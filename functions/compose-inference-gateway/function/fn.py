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

"""Compose the control plane routing gateway.

This function installs Traefik Proxy on the control plane cluster via Helm,
creates a GatewayClass and Gateway for unified endpoint routing, and
optionally installs MetalLB for kind/bare-metal clusters. The gateway address
is surfaced in status for compose-model-deployment to use.

Traefik is used (instead of e.g. Envoy Gateway) because it supports
per-backendRef URLRewrite filters. This is a Gateway API Extended feature
that allows each backend in a weighted traffic split to have its own path
rewrite, which Modelplane needs to route across endpoints with different
path conventions (e.g. a self-hosted model at /v1/ alongside Groq at
/openai/v1/). Envoy Gateway does not support this — see
envoyproxy/gateway#7099.
"""

import pathlib

import grpc
import yaml
from crossplane.function import logging, request, resource, response
from crossplane.function.proto.v1 import run_function_pb2 as fnv1
from crossplane.function.proto.v1 import run_function_pb2_grpc as grpcv1
from models.ai.modelplane.inferencegateway import v1alpha1
from models.io.crossplane.m.helm.release import v1beta1 as helmv1beta1
from models.io.k8s.apimachinery.pkg.apis.meta import v1 as metav1

_HERE = pathlib.Path(__file__).parent

# Gateway API CRDs (standard channel, v1.5.1) vendored from upstream:
# https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.1/standard-install.yaml
#
# Traefik's Helm chart does not ship the Gateway API CRDs, and on a fresh
# control plane nothing else installs them, so the Traefik release fails to
# render its GatewayClass. We compose the CRDs directly onto the control
# plane before the Traefik release. v1.5.1 is the version Traefik v3.7
# supports, and its standard channel serves TLSRoute and BackendTLSPolicy as
# v1, which Traefik watches.
#
# We install only the CustomResourceDefinitions, not the
# ValidatingAdmissionPolicy ("safe-upgrades") that the upstream bundle also
# ships. Composing a policy that governs CRD writes alongside the very CRD
# writes it governs is needlessly fragile.
_GATEWAY_API_CRDS = [
    doc
    for doc in yaml.safe_load_all((_HERE / "gateway_api_crds.yaml").read_text())
    if doc and doc.get("kind") == "CustomResourceDefinition"
]

# Condition types and reasons for the InferenceGateway XR.
CONDITION_TYPE_CONTROLLER_READY = "ControllerReady"

CONDITION_REASON_CONTROLLER_HEALTHY = "ControllerHealthy"
CONDITION_REASON_INSTALLING = "Installing"

# ProviderConfig name for in-cluster Helm releases on the control plane.
_PC_NAME = "modelplane-in-cluster"

# The modelplane-system namespace. Used for Helm release metadata,
# Usage resources, and the Gateway resource.
_NAMESPACE_SYSTEM = "modelplane-system"

# The control plane gateway name. Used as the Gateway resource name
# and the MetalLB IP pool / L2Advertisement name.
_GATEWAY_NAME = "modelplane"

# Traefik Helm chart coordinates.
_TRAEFIK_CHART = "traefik"
_TRAEFIK_REPO = "https://traefik.github.io/charts"
_TRAEFIK_NAMESPACE = "traefik-system"
_TRAEFIK_SERVICE_NAME = "traefik"

# Traefik's GatewayClass controllerName and the GatewayClass name we
# compose for it.
_TRAEFIK_GATEWAY_CLASS = "traefik"
_TRAEFIK_CONTROLLER_NAME = "traefik.io/gateway-controller"

# Traefik's default "web" entryPoint listens on this port internally.
# The Gateway listener port must match the entryPoint's internal port,
# not the Service's exposed port. The Helm chart exposes the same
# entryPoint at port 80 on the Service by default.
_TRAEFIK_WEB_ENTRYPOINT_PORT = 8000


def _crd_key(doc: dict) -> str:
    """Stable composed-resource key for a Gateway API CRD."""
    name = doc["metadata"]["name"]
    return f"gateway-api-crd-{name}"


def _helm_release(
    chart: str,
    repo: str,
    version: str,
    namespace: str,
    provider_config: str,
    values: dict | None = None,
    metadata_namespace: str | None = None,
) -> helmv1beta1.Release:
    """Build a Helm Release targeting a remote (or local) cluster.

    Args:
        chart: The Helm chart name.
        repo: The chart repository URL.
        version: The chart version.
        namespace: The namespace to install the chart into on the target cluster.
        provider_config: Name of the ProviderConfig to use.
        values: Optional Helm values dict.
        metadata_namespace: Optional namespace for the Release resource itself.
            Set this explicitly when composing from a cluster-scoped XR, since
            cluster-scoped XRs don't auto-populate namespace on composed
            namespaced resources.
    """
    md = metav1.ObjectMeta(namespace=metadata_namespace) if metadata_namespace else None

    release = helmv1beta1.Release(
        metadata=md,
        spec=helmv1beta1.Spec(
            providerConfigRef=helmv1beta1.ProviderConfigRef(
                kind="ProviderConfig",
                name=provider_config,
            ),
            forProvider=helmv1beta1.ForProvider(
                chart=helmv1beta1.Chart(
                    name=chart,
                    repository=repo,
                    version=version,
                ),
                namespace=namespace,
            ),
        ),
    )
    if values:
        release.spec.forProvider.values = values
    return release


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

        # This composition declares its ordering rather than enacting it: it
        # composes every resource on every pass and lets Crossplane sequence
        # them. A Crossplane that ignores dependencies would install Traefik
        # before the Gateway API CRDs exist and uninstall it before the
        # GatewayClass it holds a finalizer on, wedging deletion. Fail the
        # pipeline instead.
        if not request.has_capability(req, fnv1.CAPABILITY_DEPENDENCIES):
            response.fatal(
                rsp,
                "Crossplane does not support composed resource dependencies, "
                "which this composition requires to order the gateway",
            )
            return rsp
        c = Composer(req, rsp)
        c.compose()
        return rsp


class Composer:
    def __init__(self, req: fnv1.RunFunctionRequest, rsp: fnv1.RunFunctionResponse) -> None:
        self.req = req
        self.rsp = rsp
        self.xr = v1alpha1.InferenceGateway(**resource.struct_to_dict(req.observed.composite.resource))

    def compose(self) -> None:
        self.compose_provider_config()
        self.compose_gateway_api_crds()
        self.compose_metallb()
        self.compose_traefik()
        self.compose_gateway()
        self.compose_dependencies()
        self.write_status()
        self.derive_conditions()

    def compose_provider_config(self) -> None:
        """Namespaced ProviderConfig for provider-helm targeting the control
        plane using the pod's own service account (in-cluster identity).
        Namespaced (not ClusterProviderConfig) so it can be named directly by
        the resources that depend on it.

        Ready only once observed. Everything that targets this ProviderConfig
        depends on it, so its readiness is what releases the rest of the
        graph: calling it ready while Crossplane has yet to persist it would
        let provider-helm act on Releases pointing at a ProviderConfig that
        doesn't exist."""
        resource.update(
            self.rsp.desired.resources["provider-config-helm"],
            {
                "apiVersion": "helm.m.crossplane.io/v1beta1",
                "kind": "ProviderConfig",
                "metadata": {"name": _PC_NAME, "namespace": _NAMESPACE_SYSTEM},
                "spec": {"credentials": {"source": "InjectedIdentity"}},
            },
        )
        if "provider-config-helm" in self.req.observed.resources:
            self.rsp.desired.resources["provider-config-helm"].ready = fnv1.READY_TRUE

    def compose_gateway_api_crds(self) -> None:
        """Compose the Gateway API CRDs onto the control plane.

        These must exist before the Traefik release renders its resources and
        before Traefik watches the Gateway API types."""
        for doc in _GATEWAY_API_CRDS:
            key = _crd_key(doc)
            resource.update(self.rsp.desired.resources[key], doc)
            if resource.get_condition(self.req.observed.resources.get(key), "Established").status == "True":
                self.rsp.desired.resources[key].ready = fnv1.READY_TRUE

    def compose_metallb(self) -> None:
        """Optional MetalLB for kind/bare-metal clusters that don't have a
        cloud load balancer controller to assign Gateway addresses."""
        t = self.xr.spec.traefik
        if not (t and t.loadBalancer == "MetalLB" and t.metallb and t.metallb.addressPool):
            return

        metallb_ns = "metallb-system"

        resource.update(
            self.rsp.desired.resources["namespace-metallb"],
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": metallb_ns},
            },
        )
        self.rsp.desired.resources["namespace-metallb"].ready = fnv1.READY_TRUE

        resource.update(
            self.rsp.desired.resources["metallb"],
            _helm_release(
                chart="metallb",
                repo="https://metallb.github.io/metallb",
                version="0.14.9",
                namespace=metallb_ns,
                provider_config=_PC_NAME,
                metadata_namespace=_NAMESPACE_SYSTEM,
            ),
        )

        resource.update(
            self.rsp.desired.resources["metallb-pool"],
            {
                "apiVersion": "metallb.io/v1beta1",
                "kind": "IPAddressPool",
                "metadata": {"name": _GATEWAY_NAME, "namespace": metallb_ns},
                "spec": {"addresses": [t.metallb.addressPool]},
            },
        )
        self.rsp.desired.resources["metallb-pool"].ready = fnv1.READY_TRUE

        resource.update(
            self.rsp.desired.resources["metallb-l2"],
            {
                "apiVersion": "metallb.io/v1beta1",
                "kind": "L2Advertisement",
                "metadata": {"name": _GATEWAY_NAME, "namespace": metallb_ns},
                "spec": {"ipAddressPools": [_GATEWAY_NAME]},
            },
        )
        self.rsp.desired.resources["metallb-l2"].ready = fnv1.READY_TRUE

    def compose_traefik(self) -> None:
        """Compose Traefik Proxy.

        It depends on the ProviderConfig, so provider-helm can act on the
        Release, and on the Gateway API CRDs, so the release can render its
        resources and Traefik can watch the Gateway API types without
        erroring. Both are declared in compose_dependencies rather than
        withheld here."""
        resource.update(
            self.rsp.desired.resources["traefik"],
            _helm_release(
                chart=_TRAEFIK_CHART,
                repo=_TRAEFIK_REPO,
                version=self.xr.spec.traefik.version,  # ty: ignore[unresolved-attribute]  # XRD guarantees traefik when backend is Traefik, the only backend
                namespace=_TRAEFIK_NAMESPACE,
                provider_config=_PC_NAME,
                values={
                    "providers": {
                        "kubernetesGateway": {
                            "enabled": True,
                            "statusAddress": {
                                "service": {
                                    "namespace": _TRAEFIK_NAMESPACE,
                                    "name": _TRAEFIK_SERVICE_NAME,
                                },
                            },
                        },
                        "kubernetesIngress": {"enabled": False},
                    },
                    # Give the Traefik Service a predictable name so
                    # statusAddress.service can reference it. The default
                    # name includes Crossplane's generated release name.
                    "service": {"nameOverride": _TRAEFIK_SERVICE_NAME},
                    # Disable Traefik's built-in Gateway creation. Crossplane
                    # composes the Gateway so it appears in observed resources
                    # and we can read status.addresses.
                    "gateway": {"enabled": False},
                    # Disable the chart's GatewayClass too. The chart renders
                    # it even when gateway.enabled is false; Crossplane
                    # composes its own GatewayClass instead.
                    "gatewayClass": {"enabled": False},
                },
                metadata_namespace=_NAMESPACE_SYSTEM,
            ),
        )

    def compose_gateway(self) -> None:
        """Compose GatewayClass and Gateway.

        Both wait on Traefik through compose_dependencies: the GatewayClass on
        the release that runs its controller, and the Gateway on the
        GatewayClass."""
        resource.update(
            self.rsp.desired.resources["gateway-class"],
            {
                "apiVersion": "gateway.networking.k8s.io/v1",
                "kind": "GatewayClass",
                "metadata": {"name": _TRAEFIK_GATEWAY_CLASS},
                "spec": {
                    "controllerName": _TRAEFIK_CONTROLLER_NAME,
                },
            },
        )

        # The Gateway listener port must match Traefik's "web" entryPoint
        # internal port, not the Service's exposed port.
        resource.update(
            self.rsp.desired.resources["gateway"],
            {
                "apiVersion": "gateway.networking.k8s.io/v1",
                "kind": "Gateway",
                "metadata": {
                    "name": _GATEWAY_NAME,
                    "namespace": _NAMESPACE_SYSTEM,
                },
                "spec": {
                    "gatewayClassName": _TRAEFIK_GATEWAY_CLASS,
                    "listeners": [
                        {
                            "name": "web",
                            "protocol": "HTTP",
                            "port": _TRAEFIK_WEB_ENTRYPOINT_PORT,
                            "allowedRoutes": {"namespaces": {"from": "All"}},
                        }
                    ],
                },
            },
        )

    def write_status(self) -> None:
        """Surface the gateway's external address. Only the address — no
        gateway-specific fields. This contract works for any routing backend."""
        status = v1alpha1.Status()

        gw_observed = self.req.observed.resources.get("gateway")
        if gw_observed:
            gw_dict = resource.struct_to_dict(gw_observed.resource)
            addresses = gw_dict.get("status", {}).get("addresses", [])
            if addresses:
                status.address = addresses[0].get("value")

        resource.update_status(self.rsp.desired.composite, status)

    def derive_conditions(self) -> None:
        """Derive readiness for all composed resources and set custom
        conditions."""
        # MetalLB readiness.
        t = self.xr.spec.traefik
        if (
            t
            and t.loadBalancer == "MetalLB"
            and t.metallb
            and t.metallb.addressPool
            and resource.get_condition(self.req.observed.resources.get("metallb"), "Ready").status == "True"
        ):
            self.rsp.desired.resources["metallb"].ready = fnv1.READY_TRUE

        # Traefik readiness.
        traefik_ready = resource.get_condition(self.req.observed.resources.get("traefik"), "Ready").status == "True"
        if traefik_ready:
            self.rsp.desired.resources["traefik"].ready = fnv1.READY_TRUE

        # ControllerReady condition.
        response.set_conditions(
            self.rsp,
            resource.Condition(
                typ=CONDITION_TYPE_CONTROLLER_READY,
                status="True" if traefik_ready else "False",
                reason=CONDITION_REASON_CONTROLLER_HEALTHY if traefik_ready else CONDITION_REASON_INSTALLING,
            ),
        )

        # GatewayClass and Gateway use Accepted (not Ready) — on kind the
        # Gateway won't be Programmed (no LoadBalancer), but Accepted means
        # the controller has scheduled it and it's usable.
        if resource.get_condition(self.req.observed.resources.get("gateway-class"), "Accepted").status == "True":
            self.rsp.desired.resources["gateway-class"].ready = fnv1.READY_TRUE

        if resource.get_condition(self.req.observed.resources.get("gateway"), "Accepted").status == "True":
            self.rsp.desired.resources["gateway"].ready = fnv1.READY_TRUE

    def compose_dependencies(self) -> None:
        """Declare the order Crossplane creates and deletes these in.

        Every edge constrains both directions at once: a resource is created
        only once what it depends on reports Ready, and what it depends on is
        deleted only once the resource is gone.

        Everything provider-helm acts on depends on the ProviderConfig, which
        keeps a Release from being created against a ProviderConfig Crossplane
        hasn't persisted and holds that ProviderConfig through teardown.
        Traefik additionally depends on the Gateway API CRDs, so its release
        can render Gateway API resources and watch those types.

        The gateway chain is the teardown case that matters. Traefik's
        controller sets a finalizer on the GatewayClass and the Gateway; if
        the release goes first, nothing is left to clear those finalizers and
        deletion wedges. Gateway -> GatewayClass -> Traefik tears down in that
        order, and brings them up in reverse, which is also the order they
        make sense in: a Gateway naming a GatewayClass no controller has
        accepted does nothing.
        """
        for key in ("metallb", "traefik"):
            if key in self.rsp.desired.resources:
                response.add_dependency(self.rsp, key, "provider-config-helm")

        if "metallb" in self.rsp.desired.resources:
            for key in ("metallb-pool", "metallb-l2"):
                response.add_dependency(self.rsp, key, "metallb")

        for doc in _GATEWAY_API_CRDS:
            response.add_dependency(self.rsp, "traefik", _crd_key(doc))

        response.add_dependency(self.rsp, "gateway-class", "traefik")
        response.add_dependency(self.rsp, "gateway", "gateway-class")
