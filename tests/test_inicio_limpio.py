"""Pruebas de inicio sin imágenes distribuidas con el repositorio."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


class TestInicioLimpio(unittest.TestCase):
    def test_ejecucion_sin_fotografias_abre_panel(self):
        with tempfile.TemporaryDirectory() as temp:
            args = main.parsear_argumentos([])
            args.datos = Path(temp)
            with patch.object(main, 'abrir_panel_visual') as ventana:
                resultado = main.ejecutar(args)
            self.assertEqual(resultado, 0)
            ventana.assert_called_once_with([], [], None)

    def test_ruta_por_defecto_es_local_y_sin_demo(self):
        args = main.parsear_argumentos([])
        self.assertEqual(args.datos, main.DATOS_REALES)
        self.assertTrue(args.ventana)
        self.assertTrue(args.graficas)


if __name__ == '__main__':
    unittest.main()
