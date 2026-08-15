from renglo.blueprint.extension_blueprints import ensure_extension_blueprints
from renglo.common import load_config


class InitializeExtension:
    """
    Per-org setup when a team is assigned to PES.
    """

    def __init__(self):
        self.config = load_config()

    def run(self, payload):
        payload = payload or {}
        org = str(payload.get("org") or "").strip()
        if not org:
            return {
                "success": False,
                "action": "initialize_extension",
                "message": "org is required",
                "input": payload,
            }

        results = [self.ensure_blueprints()]
        if not results[0].get("success"):
            return {
                "success": False,
                "action": "initialize_extension",
                "message": "PES initialization failed",
                "input": payload,
                "output": results,
            }
        return {
            "success": True,
            "action": "initialize_extension",
            "message": "PES initialized",
            "input": payload,
            "output": results,
        }

    def ensure_blueprints(self):
        return ensure_extension_blueprints(self.config, module_file=__file__)
