from PIL import Image
def main(img_dir):
    img = Image.open("images/{}.ppm".format(img_dir))
    img.save("PNGimg/{}.png".format(img_dir))
    # img.show()

if __name__ == "__main__":
    main()