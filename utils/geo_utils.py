def get_true_pixel(tile_index: int, tile_results: object, num_cols: int, result_index: int) -> tuple:
    """
    Calculate the true pixel coordinates of the center of a tile's bounding box in the entire image.

    Parameters:
        tile_index (int): The index of the tile.
        tile_results (object): The results of the prediction for the tile.
        num_rows (int): The number of rows in the image grid.

    Returns:
        tuple: The x and y coordinates of the center of the bounding box.
    """
    if tile_index < num_cols:
        rows_before_tile = 0
        cols_before_tile = tile_index
    else:
        rows_before_tile = tile_index // num_cols
        cols_before_tile = tile_index % num_cols

    x, y, width, height = tile_results.boxes.xywh[result_index].cpu()

    x_center = int(cols_before_tile * 640 + x + (width / 2))
    y_center = int(rows_before_tile * 640 + y + (height / 2))

    return x_center, y_center


def list_of_ships_and_coords(results: object, sar_img: object, n_columns: int, transformer: object) -> list:
    """
    Generate a list of ship coordinates based on the given results and SAR image.

    Args:
        results (object): The results object containing tile results.
        sar_img (object): The SAR image object.
        n_columns (int): The number of columns in the tile results.
        transformer (object): The transformer object for coordinate conversion.

    Returns:
        list: A list of dictionaries containing ship coordinates. Each dictionary has the following keys:
            - 'mmsi' (str): The ship identifier.
            - 'latitude' (float): The latitude coordinate of the ship.
            - 'longitude' (float): The longitude coordinate of the ship.
    """
    ships_and_coords = []
    counter = 1
    for tile_idx, tile_results in enumerate(results):
        for detected_ship in range(len(tile_results)):

            x_center, y_center = get_true_pixel(tile_idx, tile_results, n_columns, detected_ship)
            coordinates = sar_img.transform * (x_center, y_center)
            converted_coordinates = transformer.transform(*coordinates)
            result_dict = {
                'mmsi': f'ship_{counter}',
                'latitude': converted_coordinates[0],
                'longitude': converted_coordinates[1],
            }
            ships_and_coords.append(result_dict)
            counter += 1
    return ships_and_coords
