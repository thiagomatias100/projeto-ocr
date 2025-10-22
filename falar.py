# falar.py
import pyttsx3

def falar(msg="ler uma mensagem"):
    e = pyttsx3.init()
    # (opcional) tenta voz pt-BR se existir
    for v in e.getProperty("voices"):
        if any(k in v.name.lower() for k in ["portugu", "brazil", "pt"]):
            e.setProperty("voice", v.id); break
    e.say(msg)
    e.runAndWait()

if __name__ == "__main__":
    falar("ler uma mensagem")
