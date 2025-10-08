from PIL import Image
from utils import in_file, out_file


def esc_cinza_media(colored):
    w,h = colored.size
    img = Image.new("RGB",(w,h))

    for x in range(w):
        for y in range(h):
            pxl = colored.getpixel((x,y))
            lum=(pxl[0]+pxl[1]+pxl[2])//3
            img.putpixel((x,y),(lum,lum,lum)) 
    return img


def esc_cinza(colored):
    w,h = colored.size
    img = Image.new("RGB",(w,h))

    for x in range(w):
        for y in range(h):
            pxl = colored.getpixel((x,y))
            lum=int((0.3*pxl[0]+0.59*pxl[1]+0.11*pxl[2]))
            img.putpixel((x,y),(lum,lum,lum)) 
    return img
if __name__ == "__main__":
    img = Image.open(in_file("diplomamat.png"))
    print(img.getpixel((100,100)))
    print(img.getpixel((150,150)))
    print(img.getpixel((250,250)))
    print(img.getpixel((100,100)))
    print(img.getpixel((150,150)))
    print(img.getpixel((200,200)))

    cpf = Image.open(in_file("diplomamat.png"))
    pb_cpf = esc_cinza(cpf)
    pb_cpf.save(out_file("pb_diplomamat.png"))