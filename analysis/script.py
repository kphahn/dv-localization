import re
import csv

# Pfad zur Textdatei
file_path = "input.txt"
csv_output = "output.csv"

# Muster zum Extrahieren der gewünschten Werte
pattern = r"rmse_after:\s+(-?[\d.]+)"

# Liste zum Speichern der extrahierten Werte
extracted_data = []

# Öffnen der Textdatei und Durchlesen des Inhalts
with open("Datasets/track_s42_livox/statistics_info.txt", "r") as file:
    content = file.read()

    # Alle Übereinstimmungen des Musters finden
    matches = re.findall(pattern, content)

    # In die Liste einfügen
    for match in matches:
        extracted_data.append(match)

with open(csv_output, "w", newline="") as csvfile:
    for l in extracted_data:
        print(l, file=csvfile)
# CSV-Datei schreiben
# with open(csv_output, "w", newline="") as csvfile:
#     csvwriter = csv.writer(csvfile)

#     # Header schreiben
#     csvwriter.writerow(["rmse_after", "difference"])

#     # Zeilen mit den extrahierten Werten schreiben
#     csvwriter.writerows(extracted_data)

print(f"Daten erfolgreich extrahiert und in {csv_output} gespeichert.")
