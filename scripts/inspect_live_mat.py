# Questo script analizza i file MATLAB del dataset LIVE IQA.
# Carica i file contenenti i valori DMOS, i DMOS riallineati e i nomi delle
# immagini di riferimento, mostrando per ciascuno le chiavi disponibili,
# il tipo, la dimensione dei dati e alcuni elementi di esempio.

# Lo script viene utilizzato per verificare la struttura e il contenuto
# del dataset prima della sua integrazione nella pipeline IQA.


from scipy.io import loadmat

ROOT = "/work/cvcs2026/Cross_Entropy_Champions/datasets/LIVEIQA_release2"

FILES = [
    "dmos.mat",
    "dmos_realigned.mat",
    "refnames_all.mat"
]

for file in FILES:

    print("\n" + "=" * 60)
    print(file)
    print("=" * 60)

    data = loadmat(f"{ROOT}/{file}")

    for key in data:

        if key.startswith("__"):
            continue

        value = data[key]

        print(f"\nKEY: {key}")
        print(f"TYPE: {type(value)}")
        print(f"SHAPE: {value.shape}")

        try:
            flat = value.flatten()

            print("\nPrimi 5 elementi:")
            print(flat[:5])

        except Exception as e:
            print("Errore durante la stampa:", e)

    print("\n")