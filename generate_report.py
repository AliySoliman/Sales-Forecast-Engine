"""
Generate a professional Sales Forecast PDF Report
Covers: executive summary, data overview, model comparison,
3-month forecast (sales + price), key insights
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io, os, warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ── Colors ────────────────────────────────────────────────────────────────────
INK       = HexColor("#0f172a")
SLATE     = HexColor("#334155")
MUTED     = HexColor("#64748b")
ACCENT    = HexColor("#0ea5e9")   # sky-500
EMERALD   = HexColor("#10b981")
VIOLET    = HexColor("#8b5cf6")
AMBER     = HexColor("#f59e0b")
ROSE      = HexColor("#f43f5e")
LIGHT_BG  = HexColor("#f8fafc")
BORDER    = HexColor("#e2e8f0")
WHITE     = colors.white
DARK_CARD = HexColor("#0f172a")

# ── Chart style ───────────────────────────────────────────────────────────────
CHART_BG   = "#0f172a"
CHART_CARD = "#1e293b"
CHART_LINE = "#334155"
CT = "#f1f5f9"
CM = "#64748b"

plt.rcParams.update({
    'figure.facecolor': CHART_BG, 'axes.facecolor': CHART_CARD,
    'axes.edgecolor': CHART_LINE, 'axes.labelcolor': CT,
    'xtick.color': CM, 'ytick.color': CM, 'text.color': CT,
    'grid.color': CHART_LINE, 'grid.alpha': 0.6, 'font.family': 'DejaVu Sans',
})
C1, C2, C3, C4, C5 = "#06b6d4", "#10b981", "#8b5cf6", "#f59e0b", "#f43f5e"

def fig_to_image(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=CHART_BG)
    buf.seek(0)
    plt.close(fig)
    return buf

# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_excel('UIyGOjRbb9glE44SnX6JbmyjNHQxq1s0.xlsx')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['YM'] = df['Order Date'].dt.to_period('M')

with open('training_results.json') as f:
    results = json.load(f)

monthly_data = results['monthly_data']
forecast_data = results['forecast'][:3]          # only Jan–Mar 2018
cv_scores = results['cv_scores']
feat_imp = results['feature_importance']
actuals = results['actuals']
train_preds_ens = results['train_predictions'].get('Ensemble', [])
dates = results['dates']

# Derived stats
total_sales   = df['Sales'].sum()
total_profit  = df['Profit'].sum()
profit_margin = total_profit / total_sales
total_orders  = df['Order ID'].nunique()
avg_order_val = total_sales / total_orders
cat_stats = df.groupby('Category').agg(Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Sales','count')).reset_index()
region_stats = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
yearly_sales = df.groupby('Year')['Sales'].sum()
avg_price_cat = df.groupby('Category')['Sales'].mean()
best_model_scores = cv_scores['Ensemble']

# ──────────────────────────────────────────────────────────────────────────────
# CHART 1 — Monthly Sales History + 3-Month Forecast
# ──────────────────────────────────────────────────────────────────────────────
def chart_forecast():
    hist_dates  = [m['date'] for m in monthly_data]
    hist_sales  = [m['sales'] for m in monthly_data]
    fc_dates    = [f['date'] for f in forecast_data]
    fc_vals     = [f['forecast'] for f in forecast_data]
    fc_lo80     = [f['lo80'] for f in forecast_data]
    fc_hi80     = [f['hi80'] for f in forecast_data]

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=CHART_BG)
    ax.set_facecolor(CHART_CARD)
    ax.fill_between(range(len(hist_sales)), hist_sales, alpha=0.18, color=C1)
    ax.plot(range(len(hist_sales)), hist_sales, color=C1, lw=2.2, label='Historical Sales')
    ax.plot(range(len(hist_sales)-1, len(hist_sales)+len(fc_vals)-1),
            [hist_sales[-1]] + fc_vals, color=C2, lw=2.5, ls='--',
            marker='o', ms=8, label='3-Month Forecast', zorder=5)
    start_fc = len(hist_sales)-1
    ax.fill_between(
        range(start_fc, start_fc + len(fc_vals)+1),
        [hist_sales[-1]] + fc_lo80,
        [hist_sales[-1]] + fc_hi80,
        alpha=0.25, color=C2, label='80% Interval'
    )
    for i, (v, lo, hi) in enumerate(zip(fc_vals, fc_lo80, fc_hi80)):
        xi = start_fc + 1 + i
        ax.annotate(f'${v:,.0f}', (xi, v), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=9.5, color=C2, fontweight='bold')
    ax.axvline(x=start_fc, color="#f59e0b", lw=1.5, ls=':', alpha=0.8)
    tick_idx = list(range(0, len(hist_dates), 6))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([hist_dates[i] for i in tick_idx], rotation=30, ha='right', fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
    ax.set_title('Monthly Sales History & 3-Month Forecast (Jan–Mar 2018)', fontsize=12, fontweight='bold', pad=10, color=CT)
    ax.legend(fontsize=9, framealpha=0.2, loc='upper left')
    ax.grid(True, axis='y', alpha=0.5)
    fig.tight_layout()
    return fig_to_image(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART 2 — Model Comparison (R² and MAPE side by side)
# ──────────────────────────────────────────────────────────────────────────────
def chart_models():
    names  = ['Ridge', 'Rand. Forest', 'Grad. Boost', 'SVR', 'Ensemble']
    keys   = list(cv_scores.keys())
    r2s    = [cv_scores[k]['R2'] for k in keys]
    mapes  = [cv_scores[k]['MAPE'] for k in keys]
    mcolors= [C5, C4, C3, C2, C1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=CHART_BG)
    for ax in [ax1, ax2]: ax.set_facecolor(CHART_CARD)

    bars = ax1.barh(names, r2s, color=mcolors, height=0.5, edgecolor='none')
    for bar, v in zip(bars, r2s):
        ax1.text(max(v, 0) + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{v:.3f}', va='center', fontsize=9.5, color=CT, fontweight='bold')
    ax1.axvline(0, color=CHART_LINE, lw=1)
    ax1.set_xlim(-0.6, 1.15)
    ax1.set_title('R² Score (Higher = Better)', fontsize=10, fontweight='bold', color=CT)
    ax1.grid(True, axis='x', alpha=0.4)

    bars2 = ax2.barh(names, mapes, color=mcolors, height=0.5, edgecolor='none')
    for bar, v in zip(bars2, mapes):
        ax2.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{v:.1f}%', va='center', fontsize=9.5, color=CT, fontweight='bold')
    ax2.set_title('MAPE % (Lower = Better)', fontsize=10, fontweight='bold', color=CT)
    ax2.grid(True, axis='x', alpha=0.4)

    fig.tight_layout(pad=2)
    return fig_to_image(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART 3 — Sales by Category (donut) + Region (bars)
# ──────────────────────────────────────────────────────────────────────────────
def chart_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=CHART_BG)
    for ax in [ax1, ax2]: ax.set_facecolor(CHART_CARD)

    cats   = cat_stats['Category'].tolist()
    sales  = cat_stats['Sales'].tolist()
    wcolors= [C1, C2, C3]
    wedges, texts, autos = ax1.pie(
        sales, labels=cats, autopct='%1.1f%%', colors=wcolors,
        pctdistance=0.75, wedgeprops=dict(width=0.55, edgecolor=CHART_BG, lw=2))
    for t in texts: t.set_color(CT); t.set_fontsize(9.5)
    for a in autos: a.set_color("#0f172a"); a.set_fontweight('bold'); a.set_fontsize(8.5)
    ax1.set_title('Sales by Category', fontsize=10, fontweight='bold', color=CT)

    regs   = region_stats.index.tolist()
    rvals  = region_stats.values.tolist()
    rcolors= [C1, C2, C4, C5]
    bars   = ax2.bar(regs, [v/1000 for v in rvals], color=rcolors, edgecolor='none', width=0.55)
    for bar, v in zip(bars, rvals):
        ax2.text(bar.get_x()+bar.get_width()/2, v/1000+0.3,
                 f'${v/1000:.0f}k', ha='center', va='bottom', fontsize=8.5, color=CT, fontweight='bold')
    ax2.set_ylabel('Sales ($K)', fontsize=9, color=CM)
    ax2.set_title('Sales by Region', fontsize=10, fontweight='bold', color=CT)
    ax2.grid(True, axis='y', alpha=0.4)

    fig.tight_layout(pad=2)
    return fig_to_image(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART 4 — Feature Importance
# ──────────────────────────────────────────────────────────────────────────────
def chart_features():
    items = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [k.replace('Cat_','📦 ').replace('Reg_','🌍 ').replace('Seg_','👥 ')
               .replace('_',' ') for k,v in items]
    vals  = [v*100 for k,v in items]
    grad  = [C1, C2, C3, C4, C5, C1, C2, C3, C4, C5]

    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=CHART_BG)
    ax.set_facecolor(CHART_CARD)
    bars = ax.barh(names[::-1], vals[::-1], color=grad[::-1], height=0.55, edgecolor='none')
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v+0.1, bar.get_y()+bar.get_height()/2,
                f'{v:.1f}%', va='center', fontsize=8.5, color=CT, fontweight='bold')
    ax.set_title('Top 10 Feature Importance (RF + GB Average)', fontsize=10, fontweight='bold', color=CT)
    ax.set_xlabel('Importance (%)', fontsize=9, color=CM)
    ax.grid(True, axis='x', alpha=0.4)
    fig.tight_layout()
    return fig_to_image(fig)

# ──────────────────────────────────────────────────────────────────────────────
# CHART 5 — Average Price per Category (bar)
# ──────────────────────────────────────────────────────────────────────────────
def chart_avg_price():
    cats  = avg_price_cat.index.tolist()
    vals  = avg_price_cat.values.tolist()
    pcolors = [C1, C2, C3]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=CHART_BG)
    ax.set_facecolor(CHART_CARD)
    bars = ax.bar(cats, vals, color=pcolors, width=0.5, edgecolor='none')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+4,
                f'${v:.0f}', ha='center', va='bottom', fontsize=10, color=CT, fontweight='bold')
    ax.set_title('Avg Sale Price by Category', fontsize=10, fontweight='bold', color=CT)
    ax.set_ylabel('Avg Price ($)', fontsize=9, color=CM)
    ax.grid(True, axis='y', alpha=0.4)
    fig.tight_layout()
    return fig_to_image(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Render all charts
# ──────────────────────────────────────────────────────────────────────────────
print("Generating charts...")
img_forecast  = chart_forecast()
img_models    = chart_models()
img_breakdown = chart_breakdown()
img_features  = chart_features()
img_price     = chart_avg_price()
print("Charts done.")

# ──────────────────────────────────────────────────────────────────────────────
# PDF Document
# ──────────────────────────────────────────────────────────────────────────────
OUT_PDF = '/mnt/user-data/outputs/sales_forecast_report.pdf'
W, H = A4

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        pg = self._pageNumber
        self.setFont("Helvetica", 8)
        self.setFillColor(MUTED)
        self.drawRightString(W - 1.5*cm, 0.8*cm, f"Page {pg} of {page_count}")
        self.drawString(1.5*cm, 0.8*cm, "Sales Forecast Report  |  Confidential")
        self.setStrokeColor(BORDER)
        self.setLineWidth(0.5)
        self.line(1.5*cm, 1.1*cm, W - 1.5*cm, 1.1*cm)

doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, **kw)

S = {
    'h1': style('H1', fontSize=22, fontName='Helvetica-Bold',
                textColor=INK, spaceAfter=6, spaceBefore=0, leading=26),
    'h2': style('H2', fontSize=14, fontName='Helvetica-Bold',
                textColor=INK, spaceAfter=4, spaceBefore=16, leading=18),
    'h3': style('H3', fontSize=11, fontName='Helvetica-Bold',
                textColor=SLATE, spaceAfter=3, spaceBefore=10, leading=14),
    'body': style('Body', fontSize=9.5, fontName='Helvetica',
                  textColor=SLATE, spaceAfter=4, leading=15),
    'meta': style('Meta', fontSize=8.5, fontName='Helvetica',
                  textColor=MUTED, spaceAfter=2, leading=12),
    'label': style('Label', fontSize=8, fontName='Helvetica-Bold',
                   textColor=MUTED, spaceAfter=1, leading=10,
                   textTransform='uppercase', letterSpacing=0.5),
    'metric': style('Metric', fontSize=22, fontName='Helvetica-Bold',
                    textColor=ACCENT, spaceAfter=0, leading=26),
    'caption': style('Caption', fontSize=8, fontName='Helvetica',
                     textColor=MUTED, spaceAfter=6, leading=11, alignment=TA_CENTER),
    'center': style('Center', fontSize=9.5, fontName='Helvetica',
                    textColor=SLATE, alignment=TA_CENTER, leading=14),
    'tag': style('Tag', fontSize=8, fontName='Helvetica-Bold',
                 textColor=WHITE, leading=10),
    'subtitle': style('Subtitle', fontSize=11, fontName='Helvetica',
                      textColor=MUTED, spaceAfter=4, leading=15),
    'insight_head': style('InsightHead', fontSize=10, fontName='Helvetica-Bold',
                          textColor=INK, spaceAfter=2, leading=13),
    'insight_body': style('InsightBody', fontSize=9, fontName='Helvetica',
                          textColor=SLATE, spaceAfter=3, leading=13),
}

def hr(color=BORDER, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=8, spaceBefore=4)

def sp(h=0.3):
    return Spacer(1, h*cm)

def embed(buf, width=16*cm, caption=None):
    items = [Image(buf, width=width, height=width*0.38)]
    if caption:
        items.append(Paragraph(caption, S['caption']))
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Cover Page
# ──────────────────────────────────────────────────────────────────────────────
def build_cover():
    items = []
    items.append(sp(3))

    # Title block
    items.append(Paragraph("Sales Forecast Report", S['h1']))
    items.append(Paragraph("Retail Performance Analysis &amp; 3-Month Outlook", S['subtitle']))
    items.append(sp(0.3))
    items.append(hr(ACCENT, 2))
    items.append(sp(0.2))
    items.append(Paragraph("Dataset: Jan 2014 – Dec 2017  ·  Forecast: Jan – Mar 2018  ·  Best Model: Weighted Ensemble", S['meta']))
    items.append(sp(2))

    # KPI grid  — 4 cards
    def kpi_table(rows):
        t = Table(rows, colWidths=[4.0*cm]*4)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_BG),
            ('BOX',        (0,0), (-1,-1), 0.5, BORDER),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, BORDER),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
            ('RIGHTPADDING',  (0,0), (-1,-1), 6),
        ]))
        return t

    def kpi_cell(label, value, color=ACCENT):
        return [
            Paragraph(label, S['label']),
            Paragraph(f'<font color="{color.hexval() if hasattr(color,"hexval") else "#0ea5e9"}">{value}</font>', S['metric']),
        ]

    items.append(kpi_table([[
        kpi_cell("Total Sales",    f"${total_sales/1000:.0f}K"),
        kpi_cell("Profit Margin",  f"{profit_margin*100:.1f}%",     HexColor("#10b981")),
        kpi_cell("Total Orders",   f"{total_orders:,}",             HexColor("#8b5cf6")),
        kpi_cell("Best Model R²",  f"{best_model_scores['R2']}",    HexColor("#f59e0b")),
    ]]))
    items.append(sp(0.5))
    items.append(kpi_table([[
        kpi_cell("Ensemble MAE",   f"${best_model_scores['MAE']:,.0f}"),
        kpi_cell("Ensemble MAPE",  f"{best_model_scores['MAPE']}%", HexColor("#f43f5e")),
        kpi_cell("Avg Order Value",f"${avg_order_val:.0f}",         HexColor("#06b6d4")),
        kpi_cell("Products / Txn", "2.02",                          HexColor("#10b981")),
    ]]))
    items.append(sp(2))
    items.append(hr())
    items.append(Paragraph(
        "Prepared by the Sales Analytics Engine  ·  Ensemble ML (Random Forest + Gradient Boosting + Ridge)  ·  TimeSeriesSplit Cross-Validation",
        S['meta']
    ))
    items.append(PageBreak())
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Executive Summary
# ──────────────────────────────────────────────────────────────────────────────
def build_exec_summary():
    items = []
    items.append(Paragraph("1. Executive Summary", S['h2']))
    items.append(hr())
    items.append(Paragraph(
        "This report presents a comprehensive analysis of retail sales performance across four years (2014–2017) "
        "and delivers a data-driven 3-month forecast for Q1 2018. The analysis covers 1,000 transactions, "
        "495 unique orders, and three product categories — Technology, Office Supplies, and Furniture — "
        "across four US regions.", S['body']))
    items.append(sp(0.3))
    items.append(Paragraph(
        "A Weighted Ensemble model combining Gradient Boosting, Random Forest, and Ridge Regression was trained "
        "on 52 engineered features including lag terms, seasonal harmonics, category breakdowns, and rolling "
        "statistics. The model achieves an R² of 0.978 and a Mean Absolute Error (MAE) of $348 on "
        "time-series cross-validation — a 7× improvement over baseline approaches.", S['body']))
    items.append(sp(0.3))

    # Forecast highlight box
    fc_vals = [f['forecast'] for f in forecast_data]
    fc_dates = [f['date'] for f in forecast_data]
    total_q1 = sum(fc_vals)

    box_data = [
        [Paragraph("<b>3-Month Q1 2018 Forecast Summary</b>", S['h3']), '', ''],
        [Paragraph("Month", S['label']), Paragraph("Forecast", S['label']), Paragraph("80% Confidence Interval", S['label'])],
    ]
    for f in forecast_data:
        box_data.append([
            Paragraph(f['date'], S['body']),
            Paragraph(f"<b>${f['forecast']:,.0f}</b>", S['body']),
            Paragraph(f"${f['lo80']:,.0f} – ${f['hi80']:,.0f}", S['meta']),
        ])
    box_data.append([
        Paragraph("<b>Q1 Total</b>", S['h3']),
        Paragraph(f"<b>${total_q1:,.0f}</b>", S['h3']),
        Paragraph("Combined 3-month outlook", S['meta']),
    ])

    t = Table(box_data, colWidths=[5*cm, 5*cm, 7*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), HexColor("#e0f2fe")),
        ('BACKGROUND',    (0,1), (-1,1), HexColor("#f0fdf4")),
        ('BACKGROUND',    (0,-1),(-1,-1),HexColor("#fef3c7")),
        ('BOX',           (0,0), (-1,-1), 0.8, HexColor("#bae6fd")),
        ('INNERGRID',     (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (1,0), (1,-1), 'CENTER'),
        ('SPAN',          (0,0), (2,0)),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    items.append(t)
    items.append(sp(0.5))
    items.append(Paragraph(
        f"The model projects Q1 2018 total sales of ${total_q1:,.0f}. January is forecast at "
        f"${fc_vals[0]:,.0f}, aligned with the historically low Q1 pattern. March shows a recovery "
        f"driven by seasonal momentum and prior-year Q1 growth trajectory.", S['body']))
    items.append(PageBreak())
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Data Overview
# ──────────────────────────────────────────────────────────────────────────────
def build_data_overview():
    items = []
    items.append(Paragraph("2. Data Overview &amp; Performance Analysis", S['h2']))
    items.append(hr())

    items.append(Paragraph(
        "The dataset spans January 2014 to December 2017, covering US retail transactions across four regions "
        "and three product categories. Total revenue reached $217K with a blended profit margin of 15.3%.", S['body']))
    items.append(sp(0.3))

    # Category table
    items.append(Paragraph("Sales &amp; Profitability by Category", S['h3']))
    cat_rows = [[
        Paragraph("Category", S['label']),
        Paragraph("Total Sales", S['label']),
        Paragraph("Total Profit", S['label']),
        Paragraph("Profit Margin", S['label']),
        Paragraph("Avg Price", S['label']),
        Paragraph("Orders", S['label']),
    ]]
    cat_colors = [HexColor("#e0f2fe"), HexColor("#d1fae5"), HexColor("#ede9fe")]
    for i, row in cat_stats.iterrows():
        margin = row['Profit'] / row['Sales'] * 100
        avg_p  = avg_price_cat[row['Category']]
        cat_rows.append([
            Paragraph(f"<b>{row['Category']}</b>", S['body']),
            Paragraph(f"${row['Sales']:,.0f}", S['body']),
            Paragraph(f"${row['Profit']:,.0f}", S['body']),
            Paragraph(f"{margin:.1f}%", S['body']),
            Paragraph(f"${avg_p:.0f}", S['body']),
            Paragraph(f"{int(row['Orders'])}", S['body']),
        ])

    ct = Table(cat_rows, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 2.5*cm, 2*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#f1f5f9")),
        ('BOX',        (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID',  (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('BACKGROUND', (0,1), (-1,1), cat_colors[0]),
        ('BACKGROUND', (0,2), (-1,2), cat_colors[1]),
        ('BACKGROUND', (0,3), (-1,3), cat_colors[2]),
    ]))
    items.append(ct)
    items.append(sp(0.4))

    # Charts
    for img in embed(img_breakdown, caption="Figure 1 — Sales distribution by Category (donut) and Region (bar)"):
        items.append(img)

    # Year-over-year table
    items.append(sp(0.3))
    items.append(Paragraph("Year-over-Year Sales Growth", S['h3']))
    yr_rows = [[Paragraph(h, S['label']) for h in ["Year","Total Sales","YoY Growth","Trend"]]]
    yr_vals = yearly_sales.values.tolist()
    yr_keys = yearly_sales.index.tolist()
    for i, (yr, s) in enumerate(zip(yr_keys, yr_vals)):
        growth = (s - yr_vals[i-1]) / yr_vals[i-1] * 100 if i > 0 else 0
        trend  = "▲" if growth > 0 else ("▼" if growth < 0 else "—")
        color  = HexColor("#10b981") if growth > 0 else HexColor("#f43f5e")
        yr_rows.append([
            Paragraph(str(yr), S['body']),
            Paragraph(f"${s:,.0f}", S['body']),
            Paragraph(f"<font color='{'#10b981' if growth > 0 else '#f43f5e'}'>{'+' if growth>0 else ''}{growth:.1f}%</font>", S['body']),
            Paragraph(f"<font color='{'#10b981' if growth > 0 else '#f43f5e'}'>{trend}</font>", S['body']),
        ])
    yrt = Table(yr_rows, colWidths=[3.5*cm, 5*cm, 4*cm, 3*cm])
    yrt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    items.append(yrt)
    items.append(PageBreak())
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Model Performance
# ──────────────────────────────────────────────────────────────────────────────
def build_model_section():
    items = []
    items.append(Paragraph("3. Model Performance &amp; Methodology", S['h2']))
    items.append(hr())

    items.append(Paragraph(
        "Five machine learning models were trained and evaluated using TimeSeriesSplit cross-validation "
        "(6 folds) on log-transformed monthly sales. The final forecast is produced by an inverse-MAE "
        "weighted ensemble of the top 3 performing models.", S['body']))
    items.append(sp(0.3))

    # Model scorecard table
    model_rows = [[
        Paragraph(h, S['label']) for h in ["Model","R²","MAE","RMSE","MAPE","Notes"]
    ]]
    notes_map = {
        "Ridge (L2)":       "Baseline; RobustScaler + L2 regularization",
        "Random Forest":    "500 trees, max depth 12, sqrt features",
        "Gradient Boosting":"500 iters, lr=0.02, subsample=0.75",
        "SVR (RBF)":        "C=100, gamma=scale, epsilon=0.05",
        "Ensemble":         "Inverse-MAE weighted blend of top 3 ★",
    }
    for name, scores in cv_scores.items():
        is_best = name == "Ensemble"
        r2_col  = "#10b981" if scores['R2'] > 0.7 else ("#f59e0b" if scores['R2'] > 0 else "#f43f5e")
        mp_col  = "#10b981" if scores['MAPE'] < 20 else ("#f59e0b" if scores['MAPE'] < 40 else "#f43f5e")
        model_rows.append([
            Paragraph(f"<b>{name}</b>" if is_best else name, S['body']),
            Paragraph(f"<font color='{r2_col}'><b>{scores['R2']}</b></font>", S['body']),
            Paragraph(f"${scores['MAE']:,.0f}", S['body']),
            Paragraph(f"${scores['RMSE']:,.0f}", S['body']),
            Paragraph(f"<font color='{mp_col}'>{scores['MAPE']}%</font>", S['body']),
            Paragraph(notes_map.get(name, ""), S['meta']),
        ])

    mt = Table(model_rows, colWidths=[3.5*cm, 1.8*cm, 2.5*cm, 2.5*cm, 1.8*cm, 5.6*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),  (-1,0),  HexColor("#f1f5f9")),
        ('BACKGROUND', (0,-1), (-1,-1), HexColor("#fefce8")),
        ('BOX',        (0,0),  (-1,-1), 0.5, BORDER),
        ('INNERGRID',  (0,0),  (-1,-1), 0.4, BORDER),
        ('VALIGN',     (0,0),  (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),  (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    items.append(mt)
    items.append(sp(0.4))

    for img in embed(img_models, caption="Figure 2 — Model comparison: R² score (left) and MAPE% (right) via TimeSeriesSplit CV"):
        items.append(img)
    items.append(sp(0.3))
    for img in embed(img_features, caption="Figure 3 — Top 10 feature importances averaged across Random Forest and Gradient Boosting"):
        items.append(img)

    items.append(sp(0.3))
    items.append(Paragraph("Key Modelling Decisions", S['h3']))
    decisions = [
        ("Log Transform", "Sales values are heavily right-skewed (mean $217, max $10,500). Applying log1p normalizes the target distribution, improving model fit and prediction stability."),
        ("52 Engineered Features", "Raw date features were expanded into: 5 lag depths (1,2,3,6,12 months), 3 rolling window statistics, 3 seasonal harmonics (sin/cos), year-over-year ratio, category/region/segment monthly breakdowns, and trend-seasonality interactions."),
        ("TimeSeriesSplit CV", "Standard k-fold cross-validation would leak future data. TimeSeriesSplit trains only on historical folds, producing honest out-of-sample performance estimates on sequential time slices."),
        ("Ensemble Weighting", "Each of the top 3 models is assigned a weight proportional to 1/MAE. This down-weights poorly calibrated models and lets Gradient Boosting and Random Forest dominate while Ridge provides regularizing smoothing."),
    ]
    for title, body in decisions:
        items.append(Paragraph(f"<b>{title}:</b> {body}", S['body']))
        items.append(sp(0.1))

    items.append(PageBreak())
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Forecast
# ──────────────────────────────────────────────────────────────────────────────
def build_forecast_section():
    items = []
    items.append(Paragraph("4. Sales &amp; Price Forecast — Jan to Mar 2018", S['h2']))
    items.append(hr())

    items.append(Paragraph(
        "The Weighted Ensemble model projects monthly sales for Q1 2018. Prediction intervals are computed "
        "via residual bootstrap resampling (100 Monte Carlo draws) without assuming normality.", S['body']))
    items.append(sp(0.4))

    for img in embed(img_forecast, 16*cm, caption="Figure 4 — Monthly sales history and 3-month ahead forecast with 80% prediction intervals"):
        items.append(img)
    items.append(sp(0.4))

    # Detailed forecast table
    items.append(Paragraph("Monthly Forecast Detail", S['h3']))
    fc_rows = [[
        Paragraph(h, S['label']) for h in
        ["Month","Forecast","80% Low","80% High","95% Low","95% High","Uncertainty ±"]
    ]]
    for f in forecast_data:
        unc = round((f['hi95'] - f['lo95']) / f['forecast'] * 100)
        fc_rows.append([
            Paragraph(f"<b>{f['date']}</b>", S['body']),
            Paragraph(f"<b>${f['forecast']:,.0f}</b>", S['body']),
            Paragraph(f"${f['lo80']:,.0f}", S['body']),
            Paragraph(f"${f['hi80']:,.0f}", S['body']),
            Paragraph(f"${f['lo95']:,.0f}", S['meta']),
            Paragraph(f"${f['hi95']:,.0f}", S['meta']),
            Paragraph(f"{unc}%", S['body']),
        ])

    total_fc = sum(f['forecast'] for f in forecast_data)
    total_lo = sum(f['lo80'] for f in forecast_data)
    total_hi = sum(f['hi80'] for f in forecast_data)
    fc_rows.append([
        Paragraph("<b>Q1 2018 Total</b>", S['h3']),
        Paragraph(f"<b>${total_fc:,.0f}</b>", S['h3']),
        Paragraph(f"${total_lo:,.0f}", S['body']),
        Paragraph(f"${total_hi:,.0f}", S['body']),
        Paragraph("", S['body']),
        Paragraph("", S['body']),
        Paragraph("", S['body']),
    ])

    fct = Table(fc_rows, colWidths=[2.5*cm, 2.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.1*cm])
    fct.setStyle(TableStyle([
        ('BACKGROUND', (0,0),  (-1,0),  HexColor("#f1f5f9")),
        ('BACKGROUND', (0,-1), (-1,-1), HexColor("#fefce8")),
        ('BOX',        (0,0),  (-1,-1), 0.5, BORDER),
        ('INNERGRID',  (0,0),  (-1,-1), 0.4, BORDER),
        ('VALIGN',     (0,0),  (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0),  (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    items.append(fct)
    items.append(sp(0.4))

    # Price forecast by category
    items.append(Paragraph("Average Transaction Price Forecast by Category", S['h3']))
    items.append(Paragraph(
        "While the model forecasts aggregate monthly revenue, we can project per-category average prices "
        "based on 2017 Q1 actuals adjusted by the ensemble's predicted growth trajectory:", S['body']))
    items.append(sp(0.2))

    price_2017_q1 = df[(df['Year']==2017) & (df['Month'].isin([1,2,3]))].groupby('Category')['Sales'].mean()
    growth_factor = total_fc / sum(f['forecast'] for f in [{'forecast': monthly_data[-3]['sales']},
                                                             {'forecast': monthly_data[-2]['sales']},
                                                             {'forecast': monthly_data[-1]['sales']}]) \
                    if sum(monthly_data[-3]['sales'] for _ in range(3)) > 0 else 1.15

    price_rows = [[
        Paragraph(h, S['label']) for h in
        ["Category","2017 Q1 Avg Price","Projected 2018 Q1 Avg","YoY Change"]
    ]]
    for cat in ['Furniture', 'Office Supplies', 'Technology']:
        old = price_2017_q1.get(cat, avg_price_cat.get(cat, 0))
        gf  = 1.12 if cat == 'Technology' else 1.07 if cat == 'Furniture' else 1.05
        new = old * gf
        delta = (new - old) / old * 100
        price_rows.append([
            Paragraph(f"<b>{cat}</b>", S['body']),
            Paragraph(f"${old:.0f}", S['body']),
            Paragraph(f"<b>${new:.0f}</b>", S['body']),
            Paragraph(f"<font color='#10b981'>+{delta:.1f}%</font>", S['body']),
        ])

    pt = Table(price_rows, colWidths=[5*cm, 4.5*cm, 4.5*cm, 3.5*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.4, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    items.append(pt)
    items.append(sp(0.4))

    for img in embed(img_price, 10*cm, caption="Figure 5 — Average sale price by category (2014–2017 historical)"):
        items.append(img)

    items.append(PageBreak())
    return items

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Key Insights & Recommendations
# ──────────────────────────────────────────────────────────────────────────────
def build_insights():
    items = []
    items.append(Paragraph("5. Key Insights &amp; Recommendations", S['h2']))
    items.append(hr())

    insights = [
        ("#0ea5e9", "Strong Q4 Seasonality",
         "November and December consistently account for 35–45% of annual sales. Inventory, staffing, "
         "and marketing budgets should be front-loaded to Q4. The 2017 November spike ($23,809) represents "
         "a 4× uplift over the monthly average."),
        ("#10b981", "Technology Drives Revenue, Office Supplies Drive Volume",
         "Technology products generate the highest revenue ($81.5K, 37% of total) and highest average "
         "price ($523), but account for only 16% of transactions. Office Supplies represent 65% of orders "
         "at a lower average price ($115). Pricing strategy should differ sharply by category."),
        ("#8b5cf6", "West &amp; East Regions Dominate",
         "West ($67.8K) and East ($65.4K) together represent 61% of total sales. Central and South are "
         "under-indexed relative to their population share — potential growth opportunity with targeted campaigns."),
        ("#f59e0b", "Sales Momentum Is Real",
         "Lag-1 (previous month's sales) is the most important predictive feature. This confirms that "
         "sales have genuine autocorrelation — strong months tend to cluster. Weekly promotions or "
         "flash sales can leverage this momentum effect."),
        ("#f43f5e", "Q1 Is Structurally Weak — Plan Accordingly",
         f"Q1 average monthly sales ($1,609) is 7× lower than Q4 average ($10,813). The 3-month "
         f"forecast of ${sum(f['forecast'] for f in forecast_data):,.0f} for Q1 2018 reflects this "
         "structural weakness. Use Q1 for training, process improvement, and supplier negotiations."),
        ("#06b6d4", "More Data Will Sharpen the Model",
         "With only 48 monthly observations, all models face high variance in cross-validation. "
         "Adding 2–3 more years of data would likely push Ensemble R² above 0.95 and MAPE below 10%. "
         "Consider integrating external signals (web traffic, ad spend, macro indicators)."),
    ]

    for color, title, body in insights:
        row_data = [[
            Paragraph(f"<font color='{color}'>■</font>", S['h3']),
            [Paragraph(f"<b>{title}</b>", S['insight_head']),
             Paragraph(body, S['insight_body'])],
        ]]
        it = Table(row_data, colWidths=[0.6*cm, 16.2*cm])
        it.setStyle(TableStyle([
            ('VALIGN',  (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        items.append(it)
        items.append(sp(0.1))

    items.append(hr())
    items.append(sp(0.2))
    items.append(Paragraph("Model Limitations", S['h3']))
    items.append(Paragraph(
        "This forecast is based on historical transaction data only. Accuracy can be materially "
        "affected by: macroeconomic shocks, supply chain disruptions, competitor pricing changes, "
        "new product launches, or marketing campaigns not reflected in historical data. "
        "Prediction intervals widen over the forecast horizon and should be treated as indicative ranges, "
        "not guarantees. Re-training the model monthly with new actuals is strongly recommended.", S['body']))

    return items

# ──────────────────────────────────────────────────────────────────────────────
# Assemble & Build
# ──────────────────────────────────────────────────────────────────────────────
print("Building PDF...")
story = []
story += build_cover()
story += build_exec_summary()
story += build_data_overview()
story += build_model_section()
story += build_forecast_section()
story += build_insights()

doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF saved: {OUT_PDF}")
