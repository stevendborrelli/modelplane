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

"""Tests for the compose-inference-gateway function.

Every pass composes the same resources: the function declares the order and
Crossplane enacts it, rather than withholding resources from desired state
until their prerequisites are observed. What moves between passes is
readiness, and readiness is what releases the next resource, so the cases
below walk the readiness progression rather than a composition one.
"""

import dataclasses
import unittest

from crossplane.function import logging, resource
from crossplane.function.proto.v1 import run_function_pb2 as fnv1
from function import fn
from google.protobuf import duration_pb2 as durationpb
from google.protobuf import json_format
from google.protobuf import struct_pb2 as structpb
from models.ai.modelplane.inferencegateway import v1alpha1
from models.io.k8s.apimachinery.pkg.apis.meta import v1 as metav1


@dataclasses.dataclass
class Case:
    """A test case for compose-inference-gateway."""

    name: str
    req: fnv1.RunFunctionRequest
    want: fnv1.RunFunctionResponse


def _crd_desired_resources(ready: bool) -> dict:
    """Desired Gateway API CRD resources, built from the same vendored bundle
    the function composes so the test stays in sync. When ready is True each
    CRD is marked READY_TRUE, matching a pass where the CRDs are observed as
    Established."""
    out = {}
    for doc in fn._GATEWAY_API_CRDS:
        key = fn._crd_key(doc)
        res = fnv1.Resource(resource=resource.dict_to_struct(doc))
        if ready:
            res.ready = fnv1.READY_TRUE
        out[key] = res
    return out


def _crd_observed_resources() -> dict:
    """Observed Gateway API CRD resources, each reporting Established."""
    out = {}
    for doc in fn._GATEWAY_API_CRDS:
        key = fn._crd_key(doc)
        observed = {
            "apiVersion": doc["apiVersion"],
            "kind": doc["kind"],
            "status": {"conditions": [{"type": "Established", "status": "True"}]},
        }
        out[key] = fnv1.Resource(resource=resource.dict_to_struct(observed))
    return out


def _pc_desired(ready: bool) -> fnv1.Resource:
    """The desired ProviderConfig.

    Ready only once observed: everything provider-helm acts on depends on it,
    so its readiness is what releases the rest of the graph."""
    res = fnv1.Resource(
        resource=resource.dict_to_struct(
            {
                "apiVersion": "helm.m.crossplane.io/v1beta1",
                "kind": "ProviderConfig",
                "metadata": {
                    "name": "modelplane-in-cluster",
                    "namespace": "modelplane-system",
                },
                "spec": {"credentials": {"source": "InjectedIdentity"}},
            }
        ),
    )
    if ready:
        res.ready = fnv1.READY_TRUE
    return res


def _traefik_desired_release(ready: bool) -> fnv1.Resource:
    """The desired Traefik Helm Release."""
    res = fnv1.Resource(
        resource=resource.dict_to_struct(
            {
                "apiVersion": "helm.m.crossplane.io/v1beta1",
                "kind": "Release",
                "metadata": {"namespace": "modelplane-system"},
                "spec": {
                    "providerConfigRef": {
                        "kind": "ProviderConfig",
                        "name": "modelplane-in-cluster",
                    },
                    "forProvider": {
                        "chart": {
                            "name": "traefik",
                            "repository": "https://traefik.github.io/charts",
                            "version": "40.2.0",
                        },
                        "namespace": "traefik-system",
                        "values": {
                            "providers": {
                                "kubernetesGateway": {
                                    "enabled": True,
                                    "statusAddress": {
                                        "service": {
                                            "namespace": "traefik-system",
                                            "name": "traefik",
                                        },
                                    },
                                },
                                "kubernetesIngress": {"enabled": False},
                            },
                            "service": {"nameOverride": "traefik"},
                            "gateway": {"enabled": False},
                            "gatewayClass": {"enabled": False},
                        },
                    },
                },
            }
        ),
    )
    if ready:
        res.ready = fnv1.READY_TRUE
    return res


def _gateway_class_desired(ready: bool) -> fnv1.Resource:
    """The desired GatewayClass. Ready once observed as Accepted."""
    res = fnv1.Resource(
        resource=resource.dict_to_struct(
            {
                "apiVersion": "gateway.networking.k8s.io/v1",
                "kind": "GatewayClass",
                "metadata": {"name": "traefik"},
                "spec": {"controllerName": "traefik.io/gateway-controller"},
            }
        ),
    )
    if ready:
        res.ready = fnv1.READY_TRUE
    return res


def _gateway_desired(ready: bool) -> fnv1.Resource:
    """The desired Gateway. Ready once observed as Accepted."""
    res = fnv1.Resource(
        resource=resource.dict_to_struct(
            {
                "apiVersion": "gateway.networking.k8s.io/v1",
                "kind": "Gateway",
                "metadata": {
                    "name": "modelplane",
                    "namespace": "modelplane-system",
                },
                "spec": {
                    "gatewayClassName": "traefik",
                    "listeners": [
                        {
                            "name": "web",
                            "protocol": "HTTP",
                            "port": 8000,
                            "allowedRoutes": {"namespaces": {"from": "All"}},
                        },
                    ],
                },
            }
        ),
    )
    if ready:
        res.ready = fnv1.READY_TRUE
    return res


def _desired(*, pc_ready: bool, crds_ready: bool, traefik_ready: bool, gw_ready: bool) -> dict:
    """The full desired resource set at one point in the progression.

    It never changes shape - only readiness moves - which is the point: the
    function composes the whole graph on every pass.
    """
    return {
        "provider-config-helm": _pc_desired(ready=pc_ready),
        **_crd_desired_resources(ready=crds_ready),
        "traefik": _traefik_desired_release(ready=traefik_ready),
        "gateway-class": _gateway_class_desired(ready=gw_ready),
        "gateway": _gateway_desired(ready=gw_ready),
    }


def _dependencies() -> list[fnv1.Dependency]:
    """The ordering edges the function declares, with no MetalLB configured.

    Traefik waits on the ProviderConfig, so provider-helm can act on the
    Release, and on every Gateway API CRD, so the release can render Gateway
    API resources. The gateway chain is the teardown case: Traefik's
    controller sets finalizers on the GatewayClass and the Gateway, so it has
    to outlive both.
    """
    deps = [fnv1.Dependency(resource="traefik", composed_resource="provider-config-helm")]
    deps += [fnv1.Dependency(resource="traefik", composed_resource=fn._crd_key(doc)) for doc in fn._GATEWAY_API_CRDS]
    deps.append(fnv1.Dependency(resource="gateway-class", composed_resource="traefik"))
    deps.append(fnv1.Dependency(resource="gateway", composed_resource="gateway-class"))
    return deps


def _observed_pc() -> dict:
    """The ProviderConfig as Crossplane observes it once persisted."""
    return {
        "provider-config-helm": fnv1.Resource(
            resource=resource.dict_to_struct({"apiVersion": "helm.m.crossplane.io/v1beta1", "kind": "ProviderConfig"})
        ),
    }


def _observed_ready(*conditions: str) -> fnv1.Resource:
    """An observed resource reporting each named condition True."""
    return fnv1.Resource(
        resource=resource.dict_to_struct(
            {"status": {"conditions": [{"type": c, "status": "True"} for c in conditions]}}
        )
    )


def _request(observed: dict | None = None) -> fnv1.RunFunctionRequest:
    """A request from a Crossplane that supports dependencies."""
    return fnv1.RunFunctionRequest(
        meta=fnv1.RequestMeta(capabilities=[fnv1.CAPABILITY_CAPABILITIES, fnv1.CAPABILITY_DEPENDENCIES]),
        observed=fnv1.State(
            composite=fnv1.Resource(
                resource=resource.dict_to_struct(
                    v1alpha1.InferenceGateway(
                        metadata=metav1.ObjectMeta(
                            name="test-gateway",
                            namespace="modelplane-system",
                        ),
                        spec=v1alpha1.Spec(traefik=v1alpha1.Traefik(version="40.2.0")),
                    ).model_dump(exclude_none=True, mode="json")
                ),
            ),
            resources=observed or {},
        ),
    )


def _response(resources: dict, *, controller_ready: bool, status: dict | None = None) -> fnv1.RunFunctionResponse:
    """A whole expected response: 60s TTL, empty context, the XR status."""
    return fnv1.RunFunctionResponse(
        meta=fnv1.ResponseMeta(ttl=durationpb.Duration(seconds=60)),
        desired=fnv1.State(
            composite=fnv1.Resource(resource=resource.dict_to_struct({"status": status if status is not None else {}})),
            resources=resources,
        ),
        conditions=[
            fnv1.Condition(
                type="ControllerReady",
                status=fnv1.STATUS_CONDITION_TRUE if controller_ready else fnv1.STATUS_CONDITION_FALSE,
                reason="ControllerHealthy" if controller_ready else "Installing",
            ),
        ],
        context=structpb.Struct(),
        dependencies=fnv1.Dependencies(items=_dependencies()),
    )


def _dict(rsp: fnv1.RunFunctionResponse) -> dict:
    """A response as a dict, with its dependencies in a stable order.

    Dependencies are a repeated field, so MessageToDict preserves the order
    the function emitted them in. That order carries no meaning - Crossplane
    reads the edges as a set - so sort both sides rather than pin an emission
    order the function is free to change.
    """
    d = json_format.MessageToDict(rsp)
    items = d.get("dependencies", {}).get("items")
    if items:
        items.sort(key=lambda dep: (dep.get("resource", ""), dep.get("composedResource", "")))
    return d


def setUpModule() -> None:
    logging.configure(level=logging.Level.DISABLED)


class TestFunctionRunner(unittest.IsolatedAsyncioTestCase):
    """Tests for FunctionRunner.RunFunction."""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = fn.FunctionRunner()

    async def test_compose(self) -> None:
        """The function composes an InferenceGateway."""
        through_traefik = _crd_observed_resources() | _observed_pc() | {"traefik": _observed_ready("Ready")}

        all_accepted = through_traefik | {
            "gateway-class": _observed_ready("Accepted"),
            "gateway": fnv1.Resource(
                resource=resource.dict_to_struct(
                    {
                        "status": {
                            "conditions": [{"type": "Accepted", "status": "True"}],
                            "addresses": [{"type": "IPAddress", "value": "203.0.113.9"}],
                        }
                    }
                )
            ),
        }

        cases = [
            Case(
                name="first pass composes the whole graph, and nothing is ready yet",
                # Crossplane creates the ProviderConfig and the CRDs and holds
                # everything else until they report Ready.
                req=_request(),
                want=_response(
                    _desired(pc_ready=False, crds_ready=False, traefik_ready=False, gw_ready=False),
                    controller_ready=False,
                ),
            ),
            Case(
                name="observed provider config and established crds release traefik",
                req=_request(observed=_crd_observed_resources() | _observed_pc()),
                want=_response(
                    _desired(pc_ready=True, crds_ready=True, traefik_ready=False, gw_ready=False),
                    controller_ready=False,
                ),
            ),
            Case(
                name="traefik ready releases the gateway class, and the gateway behind it",
                req=_request(observed=through_traefik),
                want=_response(
                    _desired(pc_ready=True, crds_ready=True, traefik_ready=True, gw_ready=False),
                    controller_ready=True,
                ),
            ),
            Case(
                name="everything accepted marks the gateway ready and publishes its address",
                req=_request(observed=all_accepted),
                want=_response(
                    _desired(pc_ready=True, crds_ready=True, traefik_ready=True, gw_ready=True),
                    controller_ready=True,
                    status={"address": "203.0.113.9"},
                ),
            ),
        ]

        for case in cases:
            with self.subTest(case.name):
                got = await self.runner.RunFunction(case.req, None)
                self.assertEqual(_dict(case.want), _dict(got), "-want, +got")

    async def test_compose_without_dependency_support(self) -> None:
        """A Crossplane that ignores dependencies would install Traefik before
        the Gateway API CRDs exist, and uninstall it before the GatewayClass
        whose finalizer its controller clears, so the function fails the
        pipeline rather than composing into it."""
        req = _request()
        req.meta.ClearField("capabilities")

        got = await self.runner.RunFunction(req, None)

        self.assertEqual([fnv1.SEVERITY_FATAL], [r.severity for r in got.results])
        self.assertEqual({}, dict(got.desired.resources))
