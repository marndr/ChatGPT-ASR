import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
Root = os.getenv("ROOT_PATH")

output_file = os.path.join(Root,"results/results_clean/results_tiny/results_certain_low_confidence_words_tiny/results_GPT-3.5-Turbo_tiny/plots_certain_low_confidence_words_tiny/Wer_vs_certain_low_confidence_plot_tiny.png")

df = pd.read_csv("evaluation_results_GPT-3.5-Turbo.csv")


#with open("evaluation_results_GPT-3.5-Turbo.html", "w") as f:
    #f.write(df.to_html())
    
df.to_excel("evaluation_results_GPT-3.5-Turbo.xlsx")

plt.plot(df["Tresh"],df["WER_corrected"], "-ob")
plt.xlabel("word confidence")
plt.ylabel("Wer")
plt.yticks([6,6.5,7,7.5,8,8.5,9])
plt.title("Wer vs word confidence for GPT-3.5-Turbo in \n certain low confidence words")
plt.savefig(output_file)


