from renglo.common import load_config


class InitializeExtension:
    """
    Per-org setup when a team is assigned to PES.

    Add org-scoped steps here as needed.
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
        return {
            "success": True,
            "action": "initialize_extension",
            "message": "PES has no org initialization steps",
            "input": payload,
            "output": [],
        }
