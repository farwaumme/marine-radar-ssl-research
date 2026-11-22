# MOANA X-band preprocessing protocol (professor instructions)

The dataset author confirmed that the displayed RGB values are **not calibrated radar intensities**; original spoke intensities cannot be recovered from the proprietary Navico SDK rendering. White denotes trails; green denotes boundaries; orange denotes detected regions.

1. Use MOANA X-band **trail** images, removing white trail pixels by replacing them with black background.
2. Pair trail and no-trail images by matching timestamp-bearing filenames. Compare cleaned trail versus no-trail images **pixel by pixel**, recording mismatch rates. The claim of equivalence is a hypothesis, not a verified fact.
3. Build pseudo-intensity grayscale variants: A (green=50, orange=200), B (green=80, orange=255), C (continuous mapping based on HSV value/saturation). The professor also distinguishes dark orange (medium-high) and bright orange (highest); exact thresholds must be documented by implementation.
4. Visualize variants and inspect whether useful radar-like structures are retained for semantic segmentation.
5. Combine selected converted MOANA images with Pohang radar images and professor-provided clean Cartesian images for the specified experiment.

Do not label any grayscale values as physically measured radar intensity. Do not state that pixelwise verification or preprocessing has been performed by this package; the implementation was not among the uploaded scripts.
