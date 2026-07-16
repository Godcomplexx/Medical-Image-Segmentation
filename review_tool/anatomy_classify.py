# -*- coding: utf-8 -*-
"""
Авто-присвоение анатомического класса I/P/S/U кандидату-очагу.

Класс McDonald определяется положением очага относительно структур мозга
(P — у желудочка, U — у коры, I — в задней ямке, S — остальное). Здесь по
срезу FLAIR и боксу очага вычисляются анатомические признаки и применяется
обученный классификатор (обучен на оригинальной разметке); при сбое —
детерминированное клиническое правило. Нужен, чтобы находки 2.5D-детектора
получали предложенный класс, а не оставались «?» — доктор лишь подтверждает.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from finetune.two_stage_pipeline.anatomy import (
    FEATURE_NAMES, build_slice_context, features_from_context, classify_by_rule)

CLS2LAB = {"I": 1, "P": 2, "S": 3, "U": 4}
DEFAULT_CLF = (REPO / "finetune" / "fileStorage" / "two_stage_full" /
               "anatomy" / "classifier_original" / "anatomy_classifier.joblib")


class AnatomyClassifier:
    def __init__(self, bundle=DEFAULT_CLF):
        self.model = None
        self.names = list(FEATURE_NAMES)
        try:
            import joblib
            b = joblib.load(str(bundle))
            self.model = b["model"]
            self.names = b.get("feature_names", list(FEATURE_NAMES))
        except Exception:
            pass
        self._ctx = {}

    def _context(self, gray, key):
        c = self._ctx.get(key)
        if c is None:
            c = build_slice_context(gray)
            self._ctx[key] = c
        return c

    def classify(self, gray, box, z_norm, key) -> int:
        """Возвращает метку 1..4 (I/P/S/U) для очага box на срезе gray."""
        ctx = self._context(gray, key)
        f = features_from_context(ctx, box, z_norm=z_norm)
        if self.model is not None:
            try:
                vec = [getattr(f, n) for n in self.names]
                cls = str(self.model.predict([vec])[0])
                if cls in CLS2LAB:
                    return CLS2LAB[cls]
            except Exception:
                pass
        return CLS2LAB.get(classify_by_rule(f), 3)
