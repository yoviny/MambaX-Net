from skimage.measure import label, regionprops


def get_bounding_box(image):
    """Compute the bounding box of the largest connected region in an image.

    Args:
        image: Binary or label array to segment into connected regions.

    Returns:
        tuple: ``(minplane, minr, minc, maxplane, maxr, maxc)`` bounding box
            coordinates of the largest region by area.
    """
    labeled_image = label(image)
    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda region: region.area)
    minplane, minr, minc, maxplane, maxr, maxc = largest_region.bbox

    return minplane, minr, minc, maxplane, maxr, maxc
