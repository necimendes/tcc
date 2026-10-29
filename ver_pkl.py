import pickle
from fpdf import FPDF

with open('pareto.pkl', 'rb') as arquivo:
    dados = pickle.load(arquivo)

texto = str(dados)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Courier", size=9)
pdf.multi_cell(0, 5, texto)
pdf.output("pareto.pdf")
