import type { PredictResponse } from './api';

// Accent blue
const BLUE = '#2b6cb0';
const DARK = '#1a202c';
const GRAY = '#718096';
const SEV_COLORS: Record<string, string> = {
  mild: '#276749',
  moderate: '#744210',
  severe: '#742a2a',
};

function hexToRgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

export async function downloadDiagnosisPdf(data: PredictResponse) {
  // Dynamically import jsPDF so it's only loaded client-side
  const { jsPDF } = await import('jspdf');
  const doc = new jsPDF({ unit: 'pt', format: 'letter' });

  const PW = doc.internal.pageSize.getWidth();   // 612
  const PH = doc.internal.pageSize.getHeight();  // 792
  const ML = 45;  // margin left
  const MR = PW - 45; // margin right
  const TW = MR - ML; // text width
  let y = 50;

  const diag = data.primary_diagnosis;

  // ── Helper: check if we need a new page ─────────────────────
  function checkPage(needed = 20) {
    if (y + needed > PH - 50) {
      doc.addPage();
      y = 50;
    }
  }

  // ── Helper: draw a section heading ──────────────────────────
  function sectionHeading(text: string) {
    checkPage(30);
    doc.setFontSize(13);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...hexToRgb(BLUE));
    doc.text(text, ML, y);
    y += 4;
    doc.setDrawColor(...hexToRgb(BLUE));
    doc.setLineWidth(0.5);
    doc.line(ML, y, MR, y);
    y += 12;
    doc.setTextColor(...hexToRgb(DARK));
  }

  // ── Helper: wrapped paragraph ───────────────────────────────
  function paragraph(text: string, fontSize = 10, color = DARK, bold = false) {
    doc.setFontSize(fontSize);
    doc.setFont('helvetica', bold ? 'bold' : 'normal');
    doc.setTextColor(...hexToRgb(color));
    const lines = doc.splitTextToSize(text, TW);
    checkPage(lines.length * (fontSize + 4));
    doc.text(lines, ML, y);
    y += lines.length * (fontSize + 4) + 4;
  }

  // ── Helper: bullet item ──────────────────────────────────────
  function bullet(text: string) {
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...hexToRgb(DARK));
    const lines = doc.splitTextToSize(`• ${text}`, TW - 10);
    checkPage(lines.length * 14);
    doc.text(lines, ML + 8, y);
    y += lines.length * 14;
  }

  // ── Helper: key/value row ────────────────────────────────────
  function kvRow(key: string, value: string, valueColor = DARK) {
    checkPage(16);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...hexToRgb(BLUE));
    doc.text(`${key}:`, ML, y);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...hexToRgb(valueColor));
    doc.text(value, ML + 110, y);
    y += 16;
  }

  // ════════════════════════════════════════════════════════════
  // HEADER
  // ════════════════════════════════════════════════════════════
  doc.setFillColor(...hexToRgb('#2b6cb0'));
  doc.rect(0, 0, PW, 40, 'F');
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(255, 255, 255);
  doc.text('MediAI Diagnostic Report', ML, 26);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.text(`Generated: ${new Date().toLocaleString()}`, MR, 26, { align: 'right' });

  y = 60;

  // ════════════════════════════════════════════════════════════
  // SUMMARY BOX
  // ════════════════════════════════════════════════════════════
  doc.setFillColor(235, 244, 255);
  doc.roundedRect(ML, y, TW, 100, 6, 6, 'F');
  doc.setDrawColor(...hexToRgb('#CBD5E0'));
  doc.setLineWidth(0.5);
  doc.roundedRect(ML, y, TW, 100, 6, 6, 'S');
  y += 14;

  kvRow('Primary Diagnosis', diag.disease);
  kvRow('Confidence Score', `${diag.confidence.toFixed(1)}%`);
  const sevColor = SEV_COLORS[diag.severity.toLowerCase()] || GRAY;
  kvRow('Severity', diag.severity.charAt(0).toUpperCase() + diag.severity.slice(1), sevColor);
  kvRow('Body System', diag.body_system);
  kvRow('Model', 'TabNet (attention-based tabular deep learning)');

  y += 16;

  // ════════════════════════════════════════════════════════════
  // DESCRIPTION
  // ════════════════════════════════════════════════════════════
  sectionHeading('Description');
  paragraph(diag.description);
  y += 6;

  // ════════════════════════════════════════════════════════════
  // FEATURE IMPORTANCE
  // ════════════════════════════════════════════════════════════
  if (diag.feature_importance && diag.feature_importance.length > 0) {
    sectionHeading('AI Explainability — Key Symptoms (TabNet)');
    paragraph('Symptoms with highest attention weight during inference:', 9, GRAY);
    y += 2;
    // Table header
    checkPage(18);
    doc.setFillColor(...hexToRgb(BLUE));
    doc.rect(ML, y - 11, TW, 16, 'F');
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(255, 255, 255);
    doc.text('Symptom', ML + 6, y);
    doc.text('Contribution', MR - 6, y, { align: 'right' });
    y += 8;
    diag.feature_importance.forEach((fi, i) => {
      checkPage(16);
      doc.setFillColor(i % 2 === 0 ? 255 : 235, i % 2 === 0 ? 255 : 244, i % 2 === 0 ? 255 : 255);
      doc.rect(ML, y - 10, TW, 16, 'F');
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...hexToRgb(DARK));
      doc.text(fi.symptom, ML + 6, y);
      doc.text(`${fi.contribution.toFixed(1)}%`, MR - 6, y, { align: 'right' });
      y += 16;
    });
    y += 10;
  }

  // ════════════════════════════════════════════════════════════
  // DIFFERENTIAL DIAGNOSES
  // ════════════════════════════════════════════════════════════
  if (data.differential_diagnoses && data.differential_diagnoses.length > 0) {
    sectionHeading('Differential Diagnoses');
    checkPage(18);
    doc.setFillColor(...hexToRgb('#4A5568'));
    doc.rect(ML, y - 11, TW, 16, 'F');
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(255, 255, 255);
    doc.text('Disease', ML + 6, y);
    doc.text('Confidence', MR - 6, y, { align: 'right' });
    y += 8;
    data.differential_diagnoses.forEach((d, i) => {
      checkPage(16);
      doc.setFillColor(i % 2 === 0 ? 255 : 247, i % 2 === 0 ? 255 : 250, i % 2 === 0 ? 255 : 252);
      doc.rect(ML, y - 10, TW, 16, 'F');
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...hexToRgb(DARK));
      doc.text(d.disease, ML + 6, y);
      doc.text(`${d.confidence.toFixed(1)}%`, MR - 6, y, { align: 'right' });
      y += 16;
    });
    y += 10;
  }

  // ════════════════════════════════════════════════════════════
  // MEDICINES
  // ════════════════════════════════════════════════════════════
  sectionHeading('Recommended Pharmacology');
  diag.medicines.forEach((med) => {
    checkPage(50);
    doc.setFillColor(247, 250, 252);
    const noteLines = med.notes ? doc.splitTextToSize(`Note: ${med.notes}`, TW - 20).length : 0;
    const boxH = 38 + noteLines * 13;
    doc.roundedRect(ML, y - 12, TW, boxH, 4, 4, 'F');
    doc.setDrawColor(...hexToRgb('#CBD5E0'));
    doc.roundedRect(ML, y - 12, TW, boxH, 4, 4, 'S');

    doc.setFontSize(11);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...hexToRgb(DARK));
    doc.text(`${med.name}`, ML + 8, y);
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...hexToRgb(GRAY));
    doc.text(`(${med.type})`, ML + 8 + doc.getTextWidth(`${med.name}`) + 4, y);
    y += 14;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(...hexToRgb(DARK));
    doc.text(`Dosage: ${med.dosage}`, ML + 8, y);
    y += 13;

    if (med.notes) {
      doc.setFontSize(9);
      doc.setTextColor(...hexToRgb(GRAY));
      const nLines = doc.splitTextToSize(`Note: ${med.notes}`, TW - 20);
      doc.text(nLines, ML + 8, y);
      y += nLines.length * 13;
    }
    y += 14;
  });

  // ════════════════════════════════════════════════════════════
  // PRECAUTIONS & LIFESTYLE
  // ════════════════════════════════════════════════════════════
  const colW = (TW - 10) / 2;

  if (diag.precautions.length > 0) {
    checkPage(30);
    // Two headings side-by-side
    doc.setFontSize(12);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(...hexToRgb(BLUE));
    doc.text('Immediate Precautions', ML, y);
    doc.text('Lifestyle Modifications', ML + colW + 10, y);
    y += 4;
    doc.setDrawColor(...hexToRgb(BLUE));
    doc.setLineWidth(0.5);
    doc.line(ML, y, ML + colW, y);
    doc.line(ML + colW + 10, y, MR, y);
    y += 12;

    const maxRows = Math.max(diag.precautions.length, diag.lifestyle.length);
    for (let i = 0; i < maxRows; i++) {
      const p = diag.precautions[i] || '';
      const l = diag.lifestyle[i] || '';
      const pLines = p ? doc.splitTextToSize(`• ${p}`, colW - 6) : [];
      const lLines = l ? doc.splitTextToSize(`• ${l}`, colW - 6) : [];
      const rowH = Math.max(pLines.length, lLines.length) * 13 + 2;
      checkPage(rowH);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(...hexToRgb(DARK));
      if (pLines.length) doc.text(pLines, ML, y);
      if (lLines.length) doc.text(lLines, ML + colW + 10, y);
      y += rowH;
    }
    y += 10;
  }

  // ════════════════════════════════════════════════════════════
  // DISCLAIMER
  // ════════════════════════════════════════════════════════════
  checkPage(30);
  doc.setDrawColor(...hexToRgb(GRAY));
  doc.setLineWidth(0.4);
  doc.line(ML, y, MR, y);
  y += 10;
  doc.setFontSize(8);
  doc.setFont('helvetica', 'oblique');
  doc.setTextColor(...hexToRgb(GRAY));
  const disclaimerLines = doc.splitTextToSize(data.disclaimer, TW);
  doc.text(disclaimerLines, ML, y);

  // ── Save ─────────────────────────────────────────────────────
  const filename = `diagnosis_${diag.disease.replace(/\s+/g, '_').toLowerCase()}.pdf`;
  doc.save(filename);
}
