


def get_utm_zone(lon: float) -> int:
    """
    Computes the UTM zone from a given longitude

    @param lon: longitude in degrees
    @returns The UTM zone index
    """

    lon += 180.0
    zone = 1 + int(lon // 6)

    return zone
