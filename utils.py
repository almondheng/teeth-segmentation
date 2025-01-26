def hex_to_rgb(hex_color):
    # Remove the '#' if it's present
    hex_color = hex_color.lstrip("#")

    # Convert the hex color to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)
