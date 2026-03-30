from __future__ import annotations

from datetime import date, timedelta


def compute_easter(year: int) -> date:
    """Calcola la data di Pasqua con l'algoritmo di Gauss/Computus."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_holidays_for_year(year: int) -> set:
    """Restituisce tutte le festività italiane per un anno dato."""
    easter = compute_easter(year)
    easter_monday = easter + timedelta(days=1)
    return {
        date(year, 1, 1),    # Capodanno
        date(year, 1, 6),    # Epifania
        easter,               # Pasqua
        easter_monday,        # Pasquetta
        date(year, 4, 25),   # Liberazione
        date(year, 5, 1),    # Festa del Lavoro
        date(year, 6, 2),    # Festa della Repubblica
        date(year, 8, 15),   # Ferragosto
        date(year, 11, 1),   # Tutti i Santi
        date(year, 12, 8),   # Immacolata
        date(year, 12, 25),  # Natale
        date(year, 12, 26),  # Santo Stefano
    }


def get_holidays_for_month(year: int, month: int) -> set:
    """Restituisce le festività italiane per un mese specifico."""
    return {d for d in get_holidays_for_year(year) if d.month == month}
