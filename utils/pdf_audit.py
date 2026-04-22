"""
PDF Audit Report Generator
Creates compliance audit logs in PDF format using ReportLab
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

# Custom colors matching AgriBloom theme
AGRI_GREEN = colors.HexColor("#166534")
AGRI_GREEN_LIGHT = colors.HexColor("#22c55e")
AGRI_GREEN_BG = colors.HexColor("#f0fdf4")
AGRI_WARNING = colors.HexColor("#f59e0b")
AGRI_ERROR = colors.HexColor("#ef4444")


def _create_custom_styles() -> dict[str, ParagraphStyle]:
    """Create custom paragraph styles for the report."""
    base_styles = getSampleStyleSheet()

    custom_styles = {
        "CustomTitle": ParagraphStyle(
            "CustomTitle",
            parent=base_styles["Title"],
            fontSize=24,
            textColor=AGRI_GREEN,
            spaceAfter=20,
            alignment=1,  # Center
        ),
        "CustomSubtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base_styles["Normal"],
            fontSize=12,
            textColor=colors.gray,
            spaceAfter=30,
            alignment=1,
        ),
        "SectionHeader": ParagraphStyle(
            "SectionHeader",
            parent=base_styles["Heading2"],
            fontSize=14,
            textColor=AGRI_GREEN,
            spaceBefore=15,
            spaceAfter=10,
            borderPadding=5,
        ),
        "FieldLabel": ParagraphStyle(
            "FieldLabel",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=colors.gray,
        ),
        "FieldValue": ParagraphStyle(
            "FieldValue",
            parent=base_styles["Normal"],
            fontSize=11,
            textColor=colors.black,
            spaceBefore=2,
            spaceAfter=8,
        ),
        "WarningText": ParagraphStyle(
            "WarningText",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=AGRI_WARNING,
            backColor=colors.HexColor("#fef3c7"),
            borderPadding=8,
        ),
        "ErrorText": ParagraphStyle(
            "ErrorText",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=AGRI_ERROR,
            backColor=colors.HexColor("#fee2e2"),
            borderPadding=8,
        ),
        "SuccessText": ParagraphStyle(
            "SuccessText",
            parent=base_styles["Normal"],
            fontSize=10,
            textColor=AGRI_GREEN,
            backColor=AGRI_GREEN_BG,
            borderPadding=8,
        ),
        "Disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base_styles["Normal"],
            fontSize=8,
            textColor=colors.gray,
            spaceBefore=20,
            leading=12,
        ),
        "Footer": ParagraphStyle(
            "Footer",
            parent=base_styles["Normal"],
            fontSize=8,
            textColor=colors.gray,
            alignment=1,
        ),
    }

    return custom_styles


def generate_audit_pdf(
    audit_payload: dict[str, Any],
    output_dir: str = "models/outputs",
) -> str:
    """
    Generate a comprehensive compliance audit PDF report.

    Args:
        audit_payload: Dictionary containing audit data
        output_dir: Directory to save the PDF

    Returns:
        Path to generated PDF file
    """
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_path / f"audit_{timestamp}.pdf"

    # Create document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1*cm,
        leftMargin=1*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )

    # Get styles
    styles = _create_custom_styles()
    base_styles = getSampleStyleSheet()

    # Build content
    story = []

    # Title
    story.append(Paragraph("🌾 AgriBloom Agentic", styles["CustomTitle"]))
    story.append(Paragraph("Compliance Audit Report", styles["CustomSubtitle"]))
    story.append(Spacer(1, 10))

    # Timestamp
    report_time = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
    story.append(Paragraph(f"Generated: {report_time}", styles["FieldLabel"]))
    story.append(Spacer(1, 20))

    # Disease Detection Section
    story.append(Paragraph("📋 Disease Detection Results", styles["SectionHeader"]))

    disease = audit_payload.get("disease", "Unknown")
    disease_localized = audit_payload.get("disease_localized", disease)
    confidence = audit_payload.get("confidence", "N/A")
    crop_type = audit_payload.get("crop_type", "Unknown")

    detection_data = [
        ["Field", "Value"],
        ["Detected Condition", disease_localized],
        ["Internal Label", disease],
        ["Confidence Score", confidence],
        ["Crop Type", crop_type.title()],
        ["Language", audit_payload.get("language", "en").upper()],
    ]

    detection_table = Table(detection_data, colWidths=[3*inch, 4*inch])
    detection_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AGRI_GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BACKGROUND", (0, 1), (-1, -1), AGRI_GREEN_BG),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
    ]))
    story.append(detection_table)
    story.append(Spacer(1, 20))

    # Compliance Status Section
    story.append(Paragraph("🛡️ Compliance Status", styles["SectionHeader"]))

    compliance_allowed = audit_payload.get("compliance_allowed", True)
    risk_level = audit_payload.get("risk_level", "low")
    violations = audit_payload.get("violations", "None")

    if compliance_allowed:
        status_style = styles["SuccessText"]
        status_text = "✅ COMPLIANT - Advisory meets FSSAI/ICAR guidelines"
    else:
        status_style = styles["ErrorText"]
        status_text = "❌ NON-COMPLIANT - Advisory contains restricted substances"

    story.append(Paragraph(status_text, status_style))
    story.append(Spacer(1, 10))

    # Risk level
    risk_colors = {
        "low": AGRI_GREEN,
        "medium": AGRI_WARNING,
        "high": AGRI_ERROR,
    }
    risk_color = risk_colors.get(risk_level, colors.gray)

    story.append(Paragraph(f"<b>Risk Level:</b> {risk_level.upper()}", base_styles["Normal"]))
    story.append(Spacer(1, 5))

    if violations and violations != "None":
        story.append(Paragraph(
            f"<b>Violations Detected:</b> {violations}",
            styles["WarningText"]
        ))
    story.append(Spacer(1, 20))

    # Weather & Market Data Section
    weather = audit_payload.get("weather", {})
    market = audit_payload.get("market", {})

    if weather or market:
        story.append(Paragraph("🌤️ Context Data", styles["SectionHeader"]))

        context_data = [["Parameter", "Value"]]

        if weather:
            context_data.append(["Temperature", f"{weather.get('temp_c', 'N/A')}°C"])
            context_data.append(["Rainfall", f"{weather.get('rain_mm', 'N/A')} mm"])
            context_data.append(["Weather Source", weather.get("source", "N/A")])

        if market:
            context_data.append(["Crop", market.get("crop", "N/A").title()])
            context_data.append(["Market Price", f"₹{market.get('modal_price', 'N/A')}/quintal"])
            context_data.append(["Nearest Mandi", market.get("mandi", "N/A")])

        context_table = Table(context_data, colWidths=[3*inch, 4*inch])
        context_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), AGRI_GREEN),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.append(context_table)
        story.append(Spacer(1, 20))

    # Recommendations Section
    recommendations = audit_payload.get("recommendations", [])
    if recommendations:
        story.append(Paragraph("📋 Recommended Actions", styles["SectionHeader"]))

        for i, rec in enumerate(recommendations[:5], 1):
            story.append(Paragraph(f"{i}. {rec}", base_styles["Normal"]))
            story.append(Spacer(1, 5))

        story.append(Spacer(1, 15))

    # Disclaimers Section
    disclaimers = audit_payload.get("disclaimers", "")
    if disclaimers:
        story.append(Paragraph("⚠️ Important Disclaimers", styles["SectionHeader"]))

        if isinstance(disclaimers, str):
            disclaimer_list = disclaimers.split(" | ")
        else:
            disclaimer_list = disclaimers

        for disclaimer in disclaimer_list[:3]:
            story.append(Paragraph(f"• {disclaimer}", styles["Disclaimer"]))

    story.append(Spacer(1, 30))

    # Footer
    story.append(Paragraph(
        "—" * 50,
        styles["Footer"]
    ))
    story.append(Paragraph(
        "This report was generated by AgriBloom Agentic AI System. "
        "All recommendations must be validated with your local Krishi Vigyan Kendra (KVK).",
        styles["Disclaimer"]
    ))
    story.append(Paragraph(
        f"Report ID: AGRI-{timestamp} | ET AI Hackathon 2026",
        styles["Footer"]
    ))

    # Build PDF
    try:
        doc.build(story)
        logger.info(f"Generated audit PDF: {pdf_path}")
        return str(pdf_path)
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        raise


# Export
__all__ = ["generate_audit_pdf"]
