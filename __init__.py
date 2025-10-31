"""
ComfyUI-DyPE: Dynamic Position Extrapolation for FLUX models
"""

from .src.patch import apply_dype_to_flux


class DyPE_FLUX:
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The FLUX model to patch with DyPE."
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target image width. Must match the width of your empty latent."
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "Target image height. Must match the height of your empty latent."
                }),
                "method": (["yarn", "ntk", "base"], {
                    "default": "yarn",
                    "tooltip": "Position encoding extrapolation method (YARN recommended)."
                }),
                "enable_dype": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Enabled",
                    "label_off": "Disabled",
                    "tooltip": "Enable or disable Dynamic Position Extrapolation for RoPE."
                }),
            },
            "optional": {
                "dype_exponent": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Controls DyPE strength over time (λt). 2.0=Exponential (best for 4K+), 1.0=Linear, 0.5=Sub-linear (better for ~2K)."
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Advanced: Base shift for the noise schedule (mu). Default is 0.5."
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.15,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Advanced: Max shift for the noise schedule (mu) at high resolutions. Default is 1.15."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_dype"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "Applies DyPE (Dynamic Position Extrapolation) to a FLUX model for ultra-high-resolution generation."

    def apply_dype(self, model, width, height, method, enable_dype, dype_exponent=2.0, base_shift=0.5, max_shift=1.15):
        """
        Clones the model and applies the DyPE patch for both the noise schedule and positional embeddings.
        """
        if not hasattr(model.model, "diffusion_model") or not hasattr(model.model.diffusion_model, "pe_embedder"):
            raise ValueError("This node is only compatible with FLUX models.")
        
        patched_model = apply_dype_to_flux(model, width, height, method, enable_dype, dype_exponent, base_shift, max_shift)
        return (patched_model,)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DyPE_FLUX": DyPE_FLUX
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DyPE_FLUX": "DyPE for FLUX"
}