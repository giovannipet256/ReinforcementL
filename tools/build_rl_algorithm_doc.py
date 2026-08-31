from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import html


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "RL_Hospital_Scheduler_Guida_Algoritmo.docx"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def p(text: str = "", style: str = "BodyText") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'


def p_runs(runs: list[tuple[str, bool]], style: str = "BodyText") -> str:
    parts = [f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>']
    for text, bold in runs:
        b = "<w:b/>" if bold else ""
        parts.append(f'<w:r><w:rPr>{b}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>')
    parts.append("</w:p>")
    return "".join(parts)


def code(text: str) -> str:
    return p(text, "Code")


def h1(text: str) -> str:
    return p(text, "Heading1")


def h2(text: str) -> str:
    return p(text, "Heading2")


def h3(text: str) -> str:
    return p(text, "Heading3")


def callout(title: str, body: str) -> str:
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="B7C9DD"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="B7C9DD"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="B7C9DD"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="B7C9DD"/>'
        '</w:tblBorders><w:tblCellMar><w:top w:w="120" w:type="dxa"/><w:left w:w="160" w:type="dxa"/>'
        '<w:bottom w:w="120" w:type="dxa"/><w:right w:w="160" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="9360"/></w:tblGrid><w:tr><w:tc><w:tcPr><w:tcW w:w="9360" w:type="dxa"/>'
        '<w:shd w:fill="F4F6F9"/></w:tcPr>'
        + p_runs([(title + ". ", True), (body, False)], "CalloutText")
        + '</w:tc></w:tr></w:tbl>'
    )


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="9360" w:type="dxa"/>'
        '<w:tblInd w:w="120" w:type="dxa"/><w:tblLook w:firstRow="1" w:noHBand="1" w:noVBand="1"/>'
        '<w:tblCellMar><w:top w:w="90" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="90" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar></w:tblPr>',
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]

    def row(cells: list[str], header: bool = False) -> str:
        chunks = ["<w:tr>"]
        if header:
            chunks.append("<w:trPr><w:tblHeader/></w:trPr>")
        for idx, cell in enumerate(cells):
            fill = '<w:shd w:fill="E8EEF5"/>' if header else ""
            style = "TableHeader" if header else "TableText"
            chunks.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[idx]}" w:type="dxa"/>{fill}</w:tcPr>{p(cell, style)}</w:tc>')
        chunks.append("</w:tr>")
        return "".join(chunks)

    out.append(row(headers, True))
    for r in rows:
        out.append(row(r, False))
    out.append("</w:tbl>")
    return "".join(out)


def build_document_body() -> str:
    parts: list[str] = []
    parts.append(p("Guida rapida all'algoritmo di Reinforcement Learning", "Title"))
    parts.append(p("Hospital Scheduler - RL Shift Planner", "Subtitle"))
    parts.append(p("Documento introduttivo per comprendere gli elementi principali del modello: ambiente, azioni, masking, reward e curriculum learning.", "BodyText"))
    parts.append(callout(
        "Idea centrale",
        "Il sistema non risolve il calendario con un algoritmo deterministico classico. Simula molti mesi di pianificazione e addestra una policy RL a scegliere turni sempre migliori, usando vincoli hard, correzioni automatiche e un sistema di premi e penalita."
    ))

    parts.append(h1("1. Ambiente: environment.py"))
    parts.append(p("L'ambiente e il cuore del problema RL. In termini Gymnasium rappresenta il mondo in cui l'agente opera: definisce cosa l'agente vede, quali azioni puo proporre, come le azioni vengono trasformate in turni reali e quale reward riceve. Nel progetto e implementato dalla classe HospitalSchedulingEnv."))
    parts.append(p("Nel nostro caso un episodio corrisponde a un periodo di pianificazione, di default 30 giorni a partire da aprile 2026. A ogni giorno l'ambiente riceve una proposta di turno per ogni dipendente, verifica vincoli e copertura, aggiorna lo stato interno e restituisce il giorno successivo."))
    parts.append(h2("Cosa viene settato nell'ambiente"))
    parts.append(table(
        ["Elemento", "Configurazione nel progetto", "Perche conta"],
        [
            ["Dipendenti", "5 medici e 8 infermieri definiti in config/settings.py", "Determina la dimensione dell'azione e i fabbisogni di copertura."],
            ["Turni", "M, P, N, R come azioni base; MP, J e AP come turni speciali gestiti dall'ambiente", "Separa le decisioni ordinarie dalle correzioni o assegnazioni speciali."],
            ["Calendario", "Data iniziale, giorni episodio, weekend, festivi e prefestivi", "Permette regole diverse per giorni ordinari e giorni speciali."],
            ["Contratti", "Dirigenza 38h/settimana, Comparto 36h/settimana, riposo minimo 11h", "Guida legalita, ore settimanali e straordinario."],
            ["Copertura", "M: 1 medico e 2 infermieri; P: 1 medico e 2 infermieri; N: 1 medico e 1 infermiere", "E il vincolo operativo principale del calendario."],
            ["Storico", "Ore, ultimo turno, streak, riposi, notti, weekend e giorni speciali lavorati", "Consente di premiare rotazione e bilanciamento, non solo copertura giornaliera."],
        ],
        [1900, 3460, 4000],
    ))
    parts.append(p("L'osservazione restituita all'agente e un vettore numerico normalizzato. Contiene feature globali, come giorno e tipo di giornata, e feature per dipendente, come ore accumulate, ultimo turno, ferie, AP, riposo pianificato, preferenze e azioni legali."))

    parts.append(h1("2. Spazio delle azioni"))
    parts.append(p("Lo spazio delle azioni e MultiDiscrete([4] * N_EMPLOYEES). Significa che l'agente sceglie contemporaneamente una delle quattro azioni base per ciascun dipendente. Con 13 dipendenti, ogni step e una proposta completa di assegnazione giornaliera."))
    parts.append(table(
        ["Azione", "Turno", "Significato"],
        [
            ["0", "M", "Mattina"],
            ["1", "P", "Pomeriggio"],
            ["2", "N", "Notte"],
            ["3", "R", "Riposo"],
        ],
        [1200, 1600, 6560],
    ))
    parts.append(h2("Gestione di AP, Jolly e MP"))
    parts.append(p_runs([("AP. ", True), ("Non e una scelta libera dell'agente. L'ambiente pianifica giornate di aggiornamento professionale; se il dipendente ha AP attivo quel giorno, la proposta dell'agente viene forzata al turno AP. Se l'agente propone un turno diverso dal riposo in quella giornata, riceve una penalita di legalita.", False)]))
    parts.append(p_runs([("Jolly. ", True), ("E una risorsa esterna usata soprattutto per coprire la notte medica quando nessun medico interno puo coprirla. Non e un'azione scelta direttamente dall'agente. L'ambiente lo attiva come riparazione della copertura notturna e lo premia o penalizza a seconda che sia inevitabile, evitabile o usato nel weekend.", False)]))
    parts.append(p_runs([("MP. ", True), ("Rappresenta un turno mattina-pomeriggio che copre sia M sia P. L'agente non lo seleziona tra le quattro azioni base; l'ambiente puo attivarlo come riparazione quando mancano coperture su mattina o pomeriggio, soprattutto nei weekend. L'attivazione ha un costo di efficienza, ma puo risolvere buchi di copertura.", False)]))

    parts.append(h1("3. Masking"))
    parts.append(p("Il masking e il meccanismo che impedisce al modello di scegliere azioni non consentite. Nel progetto e esposto dal wrapper HospitalActionMasker, mentre la logica effettiva vive nel metodo action_masks() dell'ambiente e nei controlli di legalita interni."))
    parts.append(p("La maschera e un array booleano appiattito con lunghezza N_EMPLOYEES per numero azioni. Ogni valore dice se una specifica azione e disponibile per uno specifico dipendente nel giorno corrente. True significa azione consentita, False significa azione bloccata."))
    parts.append(code("Esempio concettuale: [Dr. Rossi M=True, P=True, N=False, R=True, ...]"))
    parts.append(h2("Vincoli imposti nel caso specifico"))
    parts.append(table(
        ["Vincolo", "Effetto sulla maschera"],
        [
            ["Ferie", "Se un dipendente e in ferie, l'unica azione consentita e R."],
            ["AP pianificato", "Se il dipendente ha AP nel giorno, l'agente deve proporre R; l'ambiente assegna poi AP."],
            ["Riposo pianificato", "Il riposo obbligatorio blocca M, P e N."],
            ["Legalita di turno", "Sono bloccate azioni che violano regole su ore, riposo minimo, streak e compatibilita con lo storico."],
            ["Domenica medici", "Se un medico ha lavorato il sabato, la domenica viene forzato al riposo."],
            ["Giorni speciali", "L'alternanza su weekend/festivi/prefestivi puo bloccare una ripetizione se esiste un'alternativa legale."],
        ],
        [2600, 6760],
    ))
    parts.append(p("Questo rende l'apprendimento molto piu efficiente: il modello non deve imparare da solo che certe mosse sono impossibili, perche non le vede proprio come disponibili."))

    parts.append(h1("4. Reward e penalita"))
    parts.append(p("Il reward e la funzione obiettivo del modello. Ogni giorno, ogni settimana e a fine episodio l'ambiente calcola componenti positive e negative. Queste componenti sono aggregate in categorie pesate: coverage, legality, fairness, efficiency, calendar, weekly, monthly e preference."))
    parts.append(p("Il curriculum modifica i pesi di queste categorie: nelle prime fasi il modello viene spinto soprattutto su copertura e legalita; nelle fasi avanzate aumenta il peso di fairness, calendario, vincoli settimanali e mensili."))
    parts.append(h2("Reward giornalieri principali"))
    parts.append(table(
        ["Categoria", "Reward usati"],
        [
            ["Copertura", "slot_covered, all_coverage_met, complete_daily_coverage, dirigenza_daily_coverage, dirigenza_night_covered, infermieri_daily_coverage, infermieri_night_covered, weekday_complete_coverage, coverage_first_check_pass, coverage_residual_pass, shift_proportion_all_pass"],
            ["Legalita", "riposo_rispettato, ap_rispettato, riposo_settimanale_rispettato, copertura_n_minima"],
            ["Fairness", "shift_rotation_respected, night_rotation_respected, rest_days_balanced, role_shift_balanced, weekly_rotation_balanced, weekend_work_respected"],
            ["Efficienza", "mp_weekend_bonus, jolly_weekend_bonus"],
            ["Calendario", "special_day_rotation_bonus"],
            ["Preferenze", "valid_assignment, preference_respected, justified_rest"],
        ],
        [1900, 7460],
    ))
    parts.append(h2("Penalita giornaliere principali"))
    parts.append(table(
        ["Categoria", "Penalita usate"],
        [
            ["Copertura", "slot_uncovered_medico, slot_uncovered_infermiere, dirigenza_m_uncovered, dirigenza_p_uncovered, dirigenza_n_uncovered, infermieri_m_uncovered, infermieri_p_uncovered, infermieri_n_uncovered, coverage_first_check_fail, coverage_residual_fail, shift_proportion_fail"],
            ["Legalita", "hard_override, ferie_violated, ap_violated, rest_day_violated, rest_11h_violated, hours_exceeded, copertura_n_minima_violata"],
            ["Fairness", "consecutive_same_shift, consecutive_rest, consecutive_night, role_shift_imbalance, special_day_concentration, weekly_rotation_imbalance, consecutive_weekend_work, weekday_medico_distribution"],
            ["Efficienza", "mp_activated, jolly_inevitable, jolly_avoidable"],
            ["Preferenze", "unjustified_rest, preference_violated"],
        ],
        [1900, 7460],
    ))
    parts.append(h2("Reward settimanali e mensili"))
    parts.append(p("A fine settimana l'ambiente valuta ore settimanali, bilanciamento tra ruoli, rotazione notti e deficit adattivo di copertura. A fine mese valuta completamento episodio, copertura completa, bilanciamento dei giorni speciali, equilibrio dei turni e ore mensili."))
    parts.append(p("Questa struttura e importante perche alcuni obiettivi non sono giudicabili guardando un singolo giorno. Un giorno puo essere coperto, ma il mese puo essere squilibrato; il reward mensile serve proprio a correggere questo rischio."))

    parts.append(h1("5. Curriculum learning"))
    parts.append(p("Il curriculum learning addestra il modello per livelli progressivi. Invece di partire subito dal problema piu difficile, il training introduce gradualmente ferie, criticita, weekend, festivi, fairness e stress realistico."))
    parts.append(table(
        ["Livello", "Nome", "Timesteps massimi", "Strategia"],
        [
            ["1", "Base", "200.000", "Premia molto copertura e legalita; fairness e calendario hanno peso piu basso."],
            ["2", "Ferie semplici", "300.000", "Introduce ferie meno complesse e aumenta gradualmente il peso del calendario."],
            ["3", "Ferie critiche", "400.000", "Rende piu difficili disponibilita e preferenze, con maggiore attenzione alla fairness."],
            ["4", "Weekend/Festivi/Fairness", "500.000", "Aumenta peso di fairness, calendario e vincoli settimanali."],
            ["5", "Stress realistico", "400.000", "Pesa di piu fairness, weekly, monthly ed efficiency per avvicinare il problema alla produzione."],
        ],
        [1000, 2100, 1900, 4360],
    ))
    parts.append(p("Dopo ogni chunk di addestramento il modello viene valutato. Il passaggio di livello dipende da soglie chiamate gate: reward medio minimo, massimo numero di hard override e massimo uso del Jolly. Se il gate viene superato, il training passa prima al livello successivo; altrimenti continua fino al budget del livello."))
    parts.append(table(
        ["Livello", "avg_reward minimo", "hard override max", "Jolly max"],
        [
            ["1", "5.0", "20", "12"],
            ["2", "7.0", "16", "10"],
            ["3", "8.0", "12", "8"],
            ["4", "9.0", "10", "6"],
            ["5", "10.0", "8", "5"],
        ],
        [1800, 2500, 2500, 2560],
    ))

    parts.append(h1("Sintesi operativa"))
    parts.append(p("La strategia complessiva combina tre idee. Primo: il masking rimuove le azioni impossibili prima che il modello scelga. Secondo: il reward guida il modello verso calendari coperti, legali e bilanciati. Terzo: il curriculum rende l'apprendimento graduale, evitando di chiedere subito al modello di risolvere tutte le complessita del dominio."))
    parts.append(p("Il risultato atteso non e una garanzia matematica di ottimo globale, ma una policy capace di generare rapidamente piani plausibili, valutabili e migliorabili tramite dashboard, metriche e nuove iterazioni di reward engineering."))

    return "".join(parts)


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="BodyText"><w:name w:val="Body Text"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:qFormat/><w:pPr><w:spacing w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri Light" w:hAnsi="Calibri Light"/><w:b/><w:sz w:val="52"/><w:color w:val="0B2545"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:pPr><w:spacing w:after="220"/></w:pPr><w:rPr><w:sz w:val="26"/><w:color w:val="555555"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="200"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/><w:color w:val="2E74B5"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="280" w:after="140"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/><w:color w:val="2E74B5"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1F4D78"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:pPr><w:spacing w:before="80" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/><w:color w:val="333333"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:pPr><w:spacing w:after="0" w:line="280" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableHeader"><w:name w:val="Table Header"/><w:pPr><w:spacing w:after="0" w:line="280" w:lineRule="auto"/></w:pPr><w:rPr><w:b/><w:sz w:val="20"/><w:color w:val="0B2545"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="CalloutText"><w:name w:val="Callout Text"/><w:pPr><w:spacing w:after="0" w:line="300" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="22"/><w:color w:val="0B2545"/></w:rPr></w:style>
</w:styles>"""


def document_xml() -> str:
    body = build_document_body()
    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" '
        'w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
        '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14"><w:body>'
        + body
        + sect
        + '</w:body></w:document>'
    )


def write_docx() -> None:
    OUT.parent.mkdir(exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    word_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>"""
    settings = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/></w:settings>"""
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>Guida rapida all'algoritmo di Reinforcement Learning</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application></Properties>"""

    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", word_rels)
        z.writestr("word/document.xml", document_xml())
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/settings.xml", settings)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)


if __name__ == "__main__":
    write_docx()
    print(OUT)
