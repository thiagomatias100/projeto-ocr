import cv2
from ocr_mser import ocr_with_mser
import time


#iniciar contagem de tempo de execução 
inicio = time.time()
for i in range(10**7):
    pass
#iniciar contagem de CPU de execução 
inicio = time.process_time()
for i in range(10**7):
    pass

# imagem de teste
imagem = input("ENTRE COM O ARQUIVO .PNG OU JPG:")
img = cv2.imread(imagem)  #jpg ou .png

texto, resultados = ocr_with_mser(img)

print("|°_°| O TEXTO FOI EXTRAIDO: ")
print(texto)

# Opcional: visualizar caixas detectadas
for r in resultados:
    x0,y0,x1,y1 = r["box"]
    cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 1)

cv2.imshow("Regiões MSER", img)
cv2.waitKey(0)
#cv2.destroyAllWindows() 
#Inalizar contagem da execução
fim = time.time()
print(f"|°_°| Tempo de execução: {fim - inicio:.2f} segundos")
#Medir tempo de CPU
fim = time.process_time()
print(f"|°_°| Tempo de CPU: {fim - inicio:.4f} segundos")
