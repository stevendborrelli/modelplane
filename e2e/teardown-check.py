#!/usr/bin/env python3
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

"""Tear the e2e platform down the way a user would, and check nothing is cut short.

Run after `nix run .#e2e -- --verify` has left the clusters up. Each step
deletes resources with plain `kubectl delete`, which uses background
propagation, and then watches every object the deleted XRs composed, walking
nested XRs through their composed resource references.

Issue #477 is what this checks for. With background propagation, an XR that
drops its finalizer as soon as it's deleted disappears before what it
composed, and anything waiting on it - a Usage, a dependency, the
InferenceCluster's replica guard - is released while Helm releases and
provider-kubernetes Objects are still finalizing against the cluster. They
hang, and the run reports them as stuck.

The report covers three things:

* Every XR outlived what it composed.
* Every declared dependency was respected: a resource began deleting only once
  everything depending on it was gone.
* Nothing was left behind when the step timed out.

It exits non-zero if any of them failed.
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time

STEPS = [
    (
        "model service and deployment",
        [
            ("ModelService", "modelplane.ai/v1alpha1", "ml-team", "mock"),
            ("ModelDeployment", "modelplane.ai/v1alpha1", "ml-team", "mock-demo"),
        ],
    ),
    ("inference gateway", [("InferenceGateway", "modelplane.ai/v1alpha1", "", "default")]),
    ("inference cluster", [("InferenceCluster", "modelplane.ai/v1alpha1", "", "local")]),
]


def resource_arg(kind: str, api_version: str) -> str:
    """The kubectl resource argument for a kind, qualified so it's unambiguous."""
    if "/" not in api_version:
        return kind
    group, version = api_version.split("/", 1)
    return f"{kind}.{version}.{group}"


class Obj:
    """One object in a deletion tree, and what was seen of it."""

    def __init__(self, kind: str, api_version: str, namespace: str, name: str, parent: "Obj | None" = None) -> None:
        self.kind = kind
        self.api_version = api_version
        self.namespace = namespace
        self.name = name
        self.parent = parent
        self.resource_name = ""
        self.depends_on: list[str] = []
        self.deleting_at: float | None = None
        self.gone_at: float | None = None
        self.finalizers: list[str] = []

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.api_version, self.kind, self.namespace, self.name)

    def __str__(self) -> str:
        where = f"{self.namespace}/{self.name}" if self.namespace else self.name
        rn = f" ({self.resource_name})" if self.resource_name else ""
        return f"{self.kind} {where}{rn}"


class Watcher:
    def __init__(self, context: str) -> None:
        self.context = context
        self.objs: dict[tuple[str, str, str, str], Obj] = {}
        self.start = time.monotonic()

    def kubectl(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["kubectl", "--context", self.context, *args], capture_output=True, text=True, check=False
        )

    def get(self, o: Obj) -> dict | None:
        args = ["get", resource_arg(o.kind, o.api_version), o.name, "-o", "json", "--ignore-not-found"]
        if o.namespace:
            args += ["-n", o.namespace]
        r = self.kubectl(*args)
        if r.returncode != 0:
            msg = f"kubectl get {o}: {r.stderr.strip()}"
            raise RuntimeError(msg)
        return json.loads(r.stdout) if r.stdout.strip() else None

    def add(self, o: Obj) -> Obj:
        return self.objs.setdefault(o.key, o)

    def observe(self, o: Obj, body: dict | None, now: float) -> None:
        if body is None:
            if o.gone_at is None:
                o.gone_at = now
            return

        meta = body.get("metadata", {})
        o.finalizers = meta.get("finalizers", [])
        if meta.get("deletionTimestamp") and o.deleting_at is None:
            o.deleting_at = now

        # An XR's composed resources, wherever its schema keeps them.
        spec = body.get("spec", {})
        refs = spec.get("crossplane", {}).get("resourceRefs") or spec.get("resourceRefs") or []
        for ref in refs:
            child = self.add(
                Obj(ref["kind"], ref["apiVersion"], ref.get("namespace", o.namespace), ref["name"], parent=o)
            )
            child.resource_name = ref.get("resourceName", "")
            child.depends_on = ref.get("dependsOn", [])

    def poll(self) -> None:
        now = time.monotonic() - self.start
        live = [o for o in self.objs.values() if o.gone_at is None]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for o, body in zip(live, pool.map(self.get, live), strict=True):
                self.observe(o, body, now)

    def step(self, roots: list[tuple[str, str, str, str]], timeout: float, interval: float) -> list[Obj]:
        tree = [self.add(Obj(*r)) for r in roots]
        self.poll()  # Discover the trees before deleting anything.

        for o in tree:
            args = ["delete", resource_arg(o.kind, o.api_version), o.name, "--wait=false"]
            if o.namespace:
                args += ["-n", o.namespace]
            r = self.kubectl(*args)
            print(f"  deleted {o}: {(r.stdout or r.stderr).strip()}")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if all(o.gone_at is not None for o in self.subtree(tree)):
                break
            time.sleep(interval)

        return self.subtree(tree)

    def subtree(self, roots: list[Obj]) -> list[Obj]:
        out, frontier = [], list(roots)
        while frontier:
            o = frontier.pop()
            out.append(o)
            frontier += [c for c in self.objs.values() if c.parent is o]
        return out


def report(objs: list[Obj], interval: float) -> int:
    """Print what happened to a deletion tree, and count the problems."""
    problems = 0
    t = lambda v: "-" if v is None else f"+{v:.0f}s"  # noqa: E731

    for o in sorted(objs, key=lambda o: (o.gone_at is None, o.gone_at or 0, o.deleting_at or 0)):
        print(f"    {t(o.deleting_at):>7} deleting  {t(o.gone_at):>7} gone  {o}")

    # An XR must outlive everything it composed. Polls are interval apart, so
    # an XR and its last child seen gone in the same poll can't be told apart.
    for xr in objs:
        children = [c for c in objs if c.parent is xr]
        if not children or xr.gone_at is None:
            continue
        last = [c.gone_at for c in children if c.gone_at is not None]
        if len(last) != len(children):
            continue  # Reported as left behind below.
        if xr.gone_at < max(last):
            problems += 1
            early = [str(c) for c in children if c.gone_at > xr.gone_at]
            print(
                f"  !! {xr} was gone at {t(xr.gone_at)}, "
                f"before {len(early)} of its composed resources: {', '.join(early)}"
            )
        elif xr.gone_at - max(last) < interval:
            print(f"  ~  {xr} and its last composed resource were seen gone in the same poll")

    # A resource that others depend on must not start deleting until they're
    # gone. Edges are between siblings: composed resources of one XR.
    for o in objs:
        for dep_name in o.depends_on:
            dep = next((s for s in objs if s.parent is o.parent and s.resource_name == dep_name), None)
            if dep is None or dep.deleting_at is None or o.gone_at is None:
                continue
            if dep.deleting_at < o.gone_at - interval:
                problems += 1
                print(
                    f"  !! {dep} began deleting at {t(dep.deleting_at)}, "
                    f"while {o}, which depends on it, was still there until {t(o.gone_at)}"
                )

    for o in objs:
        if o.gone_at is None:
            problems += 1
            print(f"  !! {o} was left behind, finalizers {o.finalizers}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default="kind-modelplane-e2e-local", help="the control plane cluster's kube context")
    ap.add_argument("--timeout", type=float, default=600, help="seconds each step may take")
    ap.add_argument("--interval", type=float, default=2, help="seconds between polls")
    args = ap.parse_args()

    w = Watcher(args.context)
    problems = 0
    for label, roots in STEPS:
        print(f"\n==> deleting the {label}")
        objs = w.step(roots, args.timeout, args.interval)
        problems += report(objs, args.interval)

    print(f"\n{'OK' if problems == 0 else f'{problems} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
