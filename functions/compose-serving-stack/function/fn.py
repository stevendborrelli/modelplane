# Copyright 2026 The Modelplane Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install the serving stack on a remote cluster.

The stack is a list of components fixed at build time: the function
joins the XR's cloud and stack through the stacks package (see
function/stacks/__init__.py and design/serving-stack-generation.md) and
renders each entry - a Chart as a provider-helm Release, a Manifests as
provider-kubernetes Objects - all targeting the remote cluster via
ProviderConfigs built from the XR's secrets. The reconcile path holds no
decisions of its own: every version, values block, and membership
decision was resolved where a human reviewed a diff.

Ordering derives from the data too, in both directions. A component's
depends_on edges become dependencies the function declares on its
response: Crossplane creates a component only once every dependency
reports Ready, so bring-up proceeds in dependency waves instead of
relying on Helm retrying into absent prerequisites, and tears it down
in reverse, holding a dependency until its dependents are gone. The
one hand-rendered piece is the gateway pair - the GatewayClass and
Gateway read spec.gateway, which stays per-cluster API - plus the edges
sequencing it against the Envoy Gateway release.
"""

import grpc
from crossplane.function import logging, request, resource, response
from crossplane.function.proto.v1 import run_function_pb2 as fnv1
from crossplane.function.proto.v1 import run_function_pb2_grpc as grpcv1
from models.ai.modelplane.infrastructure.servingstack import v1alpha1
from models.io.crossplane.m.helm.providerconfig import v1beta1 as helmpcv1beta1
from models.io.crossplane.m.helm.release import v1beta1 as helmv1beta1
from models.io.crossplane.m.kubernetes.object import v1alpha1 as k8sobjv1alpha1
from models.io.crossplane.m.kubernetes.providerconfig import (
    v1alpha1 as k8spcv1alpha1,
)
from models.io.k8s.apimachinery.pkg.apis.meta import v1 as metav1

from function import gateway, stacks

# Annotation provider-helm reads as the Helm release name. The stack
# lists carry the release name per Chart entry (mp-<chart>): stable
# across chart-version upgrades so provider-helm upgrades in place,
# short enough that chart-derived names stay inside the 63-character
# label limit, and mp- reserves a namespace so Modelplane can't adopt a
# same-named release a user already runs. See issue #215 and the
# design's "Ordering and identity".
_EXTERNAL_NAME_ANNOTATION = "crossplane.io/external-name"

# Secret type that names the kubeconfig entry in the XR's secrets. Every other
# entry's type is a provider identity type, which both ProviderConfigs stamp
# verbatim as their identity.type.
_SECRET_TYPE_KUBECONFIG = "Kubeconfig"

# Composed-resource keys of the two ProviderConfigs. Everything
# targeting the remote cluster depends on the one its kind reads.
_PC_KUBERNETES = "provider-config-kubernetes"
_PC_HELM = "provider-config-helm"


def _name(meta: metav1.ObjectMeta | None) -> str:
    """The object's name, always set on resources read from the API server."""
    if meta is None or meta.name is None:
        raise ValueError("metadata.name is unexpectedly absent")
    return meta.name


def _namespace(meta: metav1.ObjectMeta | None) -> str:
    """The object's namespace, always set on namespaced resources read from the API server."""
    if meta is None or meta.namespace is None:
        raise ValueError("metadata.namespace is unexpectedly absent")
    return meta.namespace


def _helm_release(chart: stacks.Chart, provider_config: str) -> helmv1beta1.Release:
    """Build a Helm Release for a Chart entry, targeting the remote cluster."""
    release = helmv1beta1.Release(
        metadata=metav1.ObjectMeta(annotations={_EXTERNAL_NAME_ANNOTATION: chart.release}),
        spec=helmv1beta1.Spec(
            providerConfigRef=helmv1beta1.ProviderConfigRef(
                kind="ProviderConfig",
                name=provider_config,
            ),
            forProvider=helmv1beta1.ForProvider(
                chart=helmv1beta1.Chart(
                    name=chart.chart,
                    repository=chart.repository,
                    version=chart.version,
                ),
                namespace=chart.namespace,
            ),
        ),
    )
    if chart.wait:
        # Helm --wait: Ready means the workloads rolled out, so the
        # install gate orders dependents on health, not deploy. The
        # default 5m can be tight for the big monitoring charts on a
        # fresh cluster pulling images.
        release.spec.forProvider.wait = True
        release.spec.forProvider.waitTimeout = "10m"
    if chart.values:
        release.spec.forProvider.values = chart.values
    return release


def _k8s_object(
    provider_config: str,
    manifest: dict,
    *,
    cel_query: str | None = None,
) -> k8sobjv1alpha1.Object:
    """Build a provider-kubernetes Object wrapping an arbitrary manifest.

    Readiness defaults to SuccessfulCreate (the Object is Ready once applied),
    which suits resources with no meaningful runtime readiness. Pass cel_query
    for an Object whose readiness must reflect a controller-populated field of
    the observed manifest - it selects the DeriveFromCelQuery policy with that
    query (see gateway.READY_CEL), which also keeps provider-kubernetes
    re-observing on its fast poll until the query passes.
    """
    obj = k8sobjv1alpha1.Object(
        spec=k8sobjv1alpha1.Spec(
            providerConfigRef=k8sobjv1alpha1.ProviderConfigRef(
                kind="ProviderConfig",
                name=provider_config,
            ),
            forProvider=k8sobjv1alpha1.ForProvider(
                manifest=manifest,
            ),
        ),
    )
    if cel_query is not None:
        obj.spec.readiness = k8sobjv1alpha1.Readiness(
            policy="DeriveFromCelQuery",
            celQuery=cel_query,
        )
    return obj


def _pc_name(xr: v1alpha1.ServingStack) -> str:
    """Derive the ProviderConfig name from the XR."""
    return resource.child_name(_name(xr.metadata), "cluster")


class FunctionRunner(grpcv1.FunctionRunnerServiceServicer):
    """A FunctionRunner handles gRPC RunFunctionRequests."""

    def __init__(self) -> None:
        """Create a new FunctionRunner."""
        self.log = logging.get_logger()

    async def RunFunction(
        self, req: fnv1.RunFunctionRequest, _: grpc.aio.ServicerContext | None
    ) -> fnv1.RunFunctionResponse:  # ty: ignore[invalid-method-override]  # the generated grpc servicer base is untyped
        """Run the function."""
        log = self.log.bind(tag=req.meta.tag)
        log.info("Running function")

        rsp = response.to(req)

        # This composition declares its ordering rather than enacting
        # it: it composes every resource on every pass and lets
        # Crossplane sequence them. A Crossplane that ignores
        # dependencies would create the whole stack at once, into
        # ProviderConfigs that may not exist yet, and tear it down in no
        # particular order. Fail the pipeline instead - a composition
        # that doesn't reconcile is easier to diagnose than a stack that
        # half installs and won't delete.
        if not request.has_capability(req, fnv1.CAPABILITY_DEPENDENCIES):
            response.fatal(
                rsp,
                "Crossplane does not support composed resource dependencies, "
                "which this composition requires to order the stack",
            )
            return rsp

        c = Composer(req, rsp)
        c.compose()
        return rsp


class Composer:
    def __init__(self, req: fnv1.RunFunctionRequest, rsp: fnv1.RunFunctionResponse) -> None:
        self.req = req
        self.rsp = rsp
        self.xr = v1alpha1.ServingStack(**resource.struct_to_dict(req.observed.composite.resource))

    def compose(self) -> None:
        self.compose_provider_configs()

        # The XRD requires and enums both fields, so the join raising means the
        # API and the stacks package disagree on a value, a broken Modelplane
        # build, not a cluster condition. Let it crash rather than dress it up
        # as a fatal result.
        components = stacks.join(self.xr.spec.cloud, self.xr.spec.stack or "Standard")

        rendered = self.compose_components(components)
        rendered += self.compose_gateway()
        self.compose_dependencies(components)
        self.write_status()
        self.mark_readiness(rendered)

    def compose_provider_configs(self) -> None:
        """Build ProviderConfigs from the XR's secrets.

        The XRD requires a Kubeconfig secret, so one is always present.
        """
        xr_secrets = self.xr.spec.secrets or []

        kubeconfig_secret = next(s for s in xr_secrets if s.type == _SECRET_TYPE_KUBECONFIG)

        # The kubeconfig provides the cluster endpoint and CA cert. If an
        # identity secret is present, it's layered on as an identity block so the
        # provider authenticates via the cloud's IAM instead of relying on
        # whatever auth is baked into the kubeconfig.
        k8s_pc_spec = k8spcv1alpha1.Spec(
            credentials=k8spcv1alpha1.Credentials(
                source="Secret",
                secretRef=k8spcv1alpha1.SecretRef(
                    name=kubeconfig_secret.name,
                    namespace=_namespace(self.xr.metadata),
                    key=kubeconfig_secret.key,
                ),
            ),
        )
        helm_pc_spec = helmpcv1beta1.Spec(
            credentials=helmpcv1beta1.Credentials(
                source="Secret",
                secretRef=helmpcv1beta1.SecretRef(
                    name=kubeconfig_secret.name,
                    namespace=_namespace(self.xr.metadata),
                    key=kubeconfig_secret.key,
                ),
            ),
        )

        identity_secret = next(
            (s for s in xr_secrets if s.type != _SECRET_TYPE_KUBECONFIG),
            None,
        )
        if identity_secret:
            # The identity entry may carry its own namespace - the Nebius
            # credential is the Secret the Nebius ClusterProviderConfig
            # references, not one in this ServingStack's namespace.
            identity_namespace = identity_secret.namespace or _namespace(self.xr.metadata)
            k8s_pc_spec.identity = k8spcv1alpha1.Identity(
                type=identity_secret.type,  # ty: ignore[invalid-argument-type]  # non-Kubeconfig types are exactly the provider identity types
                source="Secret",
                secretRef=k8spcv1alpha1.SecretRef(
                    name=identity_secret.name,
                    namespace=identity_namespace,
                    key=identity_secret.key,
                ),
            )
            helm_pc_spec.identity = helmpcv1beta1.Identity(
                type=identity_secret.type,  # ty: ignore[invalid-argument-type]  # non-Kubeconfig types are exactly the provider identity types
                source="Secret",
                secretRef=helmpcv1beta1.SecretRef(
                    name=identity_secret.name,
                    namespace=identity_namespace,
                    key=identity_secret.key,
                ),
            )

        resource.update(
            self.rsp.desired.resources[_PC_KUBERNETES],
            k8spcv1alpha1.ProviderConfig(
                metadata=metav1.ObjectMeta(name=_pc_name(self.xr)),
                spec=k8s_pc_spec,
            ),
        )

        resource.update(
            self.rsp.desired.resources[_PC_HELM],
            helmpcv1beta1.ProviderConfig(
                metadata=metav1.ObjectMeta(name=_pc_name(self.xr)),
                spec=helm_pc_spec,
            ),
        )

    def compose_components(self, components: list[stacks.Component]) -> list[str]:
        """Render every component of the joined stack.

        A Chart renders as one provider-helm Release under the entry's
        key; a Manifests entry as one provider-kubernetes Object per
        doc, keyed by stacks.components.doc_keys.

        Every component is composed on every pass. Nothing is gated
        here: compose_dependencies declares what waits for what, and
        Crossplane creates each resource once the resources it depends
        on report Ready. A Release reports Ready when Helm deploys it,
        not when its workloads run, so those waves are deploy-order, not
        health-order.

        Returns the composed-resource keys it rendered, for readiness.
        """
        pc = _pc_name(self.xr)
        rendered: list[str] = []
        for c in components:
            if isinstance(c, stacks.Chart):
                resource.update(self.rsp.desired.resources[c.key], _helm_release(c, pc))
                rendered.append(c.key)
                continue
            for key, doc in zip(stacks.components.doc_keys(c), c.manifests, strict=True):
                resource.update(
                    self.rsp.desired.resources[key],
                    _k8s_object(pc, doc, cel_query=c.ready),
                )
                rendered.append(key)
        return rendered

    def compose_dependencies(self, components: list[stacks.Component]) -> None:
        """Declare the order Crossplane creates and deletes everything in.

        Crossplane applies composed resources concurrently unless a
        function says otherwise. Every edge declared here constrains
        both directions at once: a resource is created only once what it
        depends on reports Ready, and what it depends on is deleted only
        once the resource is gone.

        Three kinds of edge. Every Release and Object depends on the
        ProviderConfig it targets, which keeps first creation from
        racing a ProviderConfig Crossplane hasn't persisted yet, and
        holds that ProviderConfig through teardown until nothing points
        at it any more. Each component depends_on edge becomes one edge
        per (dependency doc, dependent doc) pair: the kai-scheduler
        release outlives the Queue CRs whose CRD it owns, cert-manager
        outlives the Envoy Gateway release whose webhooks need it. And
        the gateway pair, which isn't stack data, is sequenced by hand
        against the controller that owns its finalizers.
        """
        docs = {c.key: stacks.components.doc_keys(c) for c in components}

        for c in components:
            pc = _PC_HELM if isinstance(c, stacks.Chart) else _PC_KUBERNETES
            for key in docs[c.key]:
                response.add_dependency(self.rsp, key, pc)
            for dep in c.depends_on:
                for dependency_key in docs[dep]:
                    for key in docs[c.key]:
                        response.add_dependency(self.rsp, key, dependency_key)

        for key, _, _ in gateway.objects(self.xr.spec.gateway):
            response.add_dependency(self.rsp, key, _PC_KUBERNETES)

        # The Envoy Gateway controller must outlive the Gateway and
        # GatewayClass it manages: they carry finalizers it has to
        # process on delete. In the other direction the GatewayClass
        # needs the CRD that release installs before it can be applied.
        response.add_dependency(self.rsp, "gateway", "gateway-class")
        response.add_dependency(self.rsp, "gateway-class", "envoy-gateway")

    def compose_gateway(self) -> list[str]:
        """Compose the GatewayClass and Gateway on the remote cluster.

        The one hand-rendered pair, from function/gateway.py: both read
        spec.gateway, which stays per-cluster API rather than stack
        data. Ordered against the ProviderConfig and the Envoy Gateway
        release by compose_dependencies, like every component.

        Returns the composed-resource keys it rendered, for readiness.
        """
        pc = _pc_name(self.xr)
        rendered: list[str] = []
        for key, manifest, cel in gateway.objects(self.xr.spec.gateway):
            resource.update(
                self.rsp.desired.resources[key],
                _k8s_object(pc, manifest, cel_query=cel),
            )
            rendered.append(key)
        return rendered

    def write_status(self) -> None:
        """Extract the gateway address from the observed Gateway Object and
        write it to the XR's status."""
        gateway_address = None
        gateway_observed = self.req.observed.resources.get("gateway")
        if gateway_observed:
            gw_dict = resource.struct_to_dict(gateway_observed.resource)
            addresses = (
                gw_dict.get("status", {})
                .get("atProvider", {})
                .get("manifest", {})
                .get("status", {})
                .get("addresses", [])
            )
            if addresses:
                gateway_address = addresses[0].get("value")

        status = v1alpha1.Status()
        if gateway_address:
            status.gateway = v1alpha1.GatewayModel(address=gateway_address)
        resource.update_status(self.rsp.desired.composite, status)

    def mark_readiness(self, rendered: list[str]) -> None:
        """Mark composed resources as ready.

        The ProviderConfigs have no readiness condition of their own,
        but they must not be ready on arrival. Everything targeting the
        remote cluster depends on one of them, so their readiness is
        what releases the rest of the graph: calling them ready while
        Crossplane has yet to persist them would let the whole stack be
        created into ProviderConfigs that don't exist, the race the
        dependency edges exist to avoid. Being observed is what makes
        them count, and it keeps the composite from reporting Ready
        before a single stack component exists. Everything rendered
        from the stack (and the gateway pair) is ready when its observed
        Ready condition is True - for Releases that's the Helm release
        deployed (its workloads rolled out, where the entry sets wait),
        for Objects the readiness policy (SuccessfulCreate, or the
        entry's CEL query).
        """
        for r in (_PC_KUBERNETES, _PC_HELM):
            if r in self.rsp.desired.resources and r in self.req.observed.resources:
                self.rsp.desired.resources[r].ready = fnv1.READY_TRUE

        for r in rendered:
            if resource.get_condition(self.req.observed.resources.get(r), "Ready").status == "True":
                self.rsp.desired.resources[r].ready = fnv1.READY_TRUE
