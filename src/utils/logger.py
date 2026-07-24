# Questo file gestisce il salvataggio dei risultati della valutazione dei modelli
# IQA. La funzione principale permette di registrare in un file CSV le informazioni
# relative al modello utilizzato, al dataset analizzato e alle metriche ottenute
# (SRCC e PLCC), creando automaticamente la cartella di log e l'intestazione del
# file quando necessario.


import csv
import os


def save_result(
        model,
        dataset,
        srcc,
        plcc,
        file_path="logs/results.csv"
):

    os.makedirs(
        "logs",
        exist_ok=True
    )


    file_exists = os.path.isfile(
        file_path
    )


    with open(
        file_path,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)


        if not file_exists:
            writer.writerow([
                "model",
                "dataset",
                "SRCC",
                "PLCC"
            ])


        writer.writerow([
            model,
            dataset,
            srcc,
            plcc
        ])