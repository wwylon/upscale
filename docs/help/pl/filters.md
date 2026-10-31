# Filtry

Opcjonalne kroki z ilością 0–100%, dostępne na karcie **Scale**:

- **Przed:** Odszumianie, usuwanie artefaktów JPEG, odszarzanie
- **Po:** Wzmocnienie krawędzi, redukcja aureoli, kolor i kontrast, ziarno filmowe

**Resetuj filtry** czyści wszystko. Filtry działają przy przebiegach Scale (pełny obraz lub pojedynczy wycinek). Porównanie w Trybie Test **nie** uruchamia wariantów filtrów — ustaw filtry na Scale po wyborze modelu. CLI: `--pre-denoise`, `--jpeg-clean`, `--deblur`, `--edge`, `--halo-reduce`, `--color-contrast`, `--grain`.
