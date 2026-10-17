# The uv workspace's Python packages, built by uv2nix from uv.lock, for
# virtualenvs that run on the build host. The function images build their own,
# per target architecture (see functions.nix).
{
  pkgs,
  self,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
}:
let
  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = self; };
in
(pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python312; }).overrideScope (
  pkgs.lib.composeManyExtensions [
    pyproject-build-systems.overlays.wheel
    (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
    # crossplane-function-sdk-python resolves from a git branch while
    # composed-resource ordering is unreleased (see the root
    # pyproject.toml), so the set builds it from source instead of
    # taking a wheel. A source build needs its backend named: the
    # lock records a package's dependencies, not its build system.
    # Drop this with the git source.
    (final: prev: {
      crossplane-function-sdk-python = prev.crossplane-function-sdk-python.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem { hatchling = [ ]; };
      });
    })
  ]
)
