import torch


class OmnivoreFeaturesExtractor(torch.nn.Module):
    """Omnivore features extractor."""

    def __init__(self, variant: str = "omnivore_swinL_imagenet21k"):
        """Create an instance of the Omnivore features extractor.

        Parameters
        ----------
        variant : str, optional
            The Omnivore variant to use, by default "omnivore_swinL_imagenet21k"
        """
        super().__init__()
        self.model = torch.hub.load("facebookresearch/omnivore:main", model=variant, force_reload=False)
        self.model.heads.video = torch.nn.Identity()
        self.model = self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Omnivore features extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input frames.

        Returns
        -------
        torch.Tensor
            Extracted features.
        """
        return self.model(x, input_type="video")
