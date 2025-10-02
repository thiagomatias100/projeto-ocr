from native_pdf import is_native_pdf, pdf_path


if __name__ == "__main__":
    if is_native_pdf == False:
        print("usaremos outro modo")
        extrator2(pdf_path)
    else:
       print("PDF Nativo ou escaneado de boa qualidade ")
       extrator(pdf_path)