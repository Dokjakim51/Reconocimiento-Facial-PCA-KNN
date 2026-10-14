from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calidad import auditar_datos
from src.imagenes import cargar_dataset, preparar_rostro_desde_gris
from src.modelo import ReconocedorEigenfaces


class TestCalidad(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.temp.name)
        self.train = self.raiz/'personas'
        self.test = self.raiz/'prueba'

    def tearDown(self):
        self.temp.cleanup()

    def guardar(self, carpeta: Path, nombre: str, semilla: int):
        carpeta.mkdir(parents=True, exist_ok=True)
        arr = np.random.default_rng(semilla).integers(0, 255, (64, 64), dtype=np.uint8)
        ruta = carpeta/nombre
        Image.fromarray(arr).save(ruta)
        return ruta

    def test_repeticiones_entrenamiento(self):
        primera = self.guardar(self.train/'Ana', '1.png', 1)
        (self.train/'Ana'/'2.png').write_bytes(primera.read_bytes())
        self.guardar(self.train/'Bruno', '1.png', 2)
        a = auditar_datos(self.train, self.test)
        self.assertEqual(len(a.repetidas_entrenamiento), 1)
        self.assertFalse(a.valido)

    def test_fuga_entre_train_y_test(self):
        primera = self.guardar(self.train/'Ana', '1.png', 1)
        (self.test/'Ana').mkdir(parents=True)
        (self.test/'Ana'/'copia.png').write_bytes(primera.read_bytes())
        a = auditar_datos(self.train, self.test)
        self.assertEqual(len(a.filtraciones), 1)

    def test_dataset_independiente(self):
        for label, base in [('Ana', 0), ('Bruno', 100)]:
            for k in range(4):
                self.guardar(self.train/label, f'{k}.png', base+k)
        a = auditar_datos(self.train, self.test)
        self.assertTrue(a.valido, a.mostrar())
        self.assertEqual(a.unicas, {'Ana':4, 'Bruno':4})


class TestModelo(unittest.TestCase):
    @staticmethod
    def dataset():
        rng = np.random.default_rng(23)
        X=[]; y=[]
        for label, centro in [('A', 0.15), ('B', 0.75), ('C', 0.45)]:
            for _ in range(7):
                X.append(np.clip(rng.normal(centro, .012, size=(16,)),0,1))
                y.append(label)
        return np.asarray(X), np.asarray(y)

    def test_predicciones_con_significado(self):
        X, y = self.dataset()
        modelo = ReconocedorEigenfaces(tamano_imagen=(4,4), componentes=8, vecinos=3)
        modelo.entrenar(X, y)
        for etiqueta, valor in [('A', .15), ('B', .75), ('C', .45)]:
            res = modelo.predecir(np.full(16,valor), permitir_desconocido=False)
            self.assertEqual(res.etiqueta, etiqueta)
            self.assertGreaterEqual(res.apoyo_vecinos, 0)
            self.assertLessEqual(res.apoyo_vecinos, 1)
        self.assertGreaterEqual(modelo.pca.n_components_, 1)

    def test_rechazo_con_umbral(self):
        X, y = self.dataset()
        modelo = ReconocedorEigenfaces(tamano_imagen=(4,4), componentes=7)
        modelo.entrenar(X,y)
        res = modelo.predecir(np.full(16, .99), umbral=1e-4)
        self.assertTrue(res.es_desconocido)
        self.assertEqual(res.etiqueta, 'Desconocido')

    def test_invalido_por_pocas_muestras(self):
        X = np.random.default_rng(1).random((4, 16))
        with self.assertRaises(ValueError):
            ReconocedorEigenfaces(tamano_imagen=(4,4)).entrenar(X, ['A','A','B','B'])

    def test_consistencia_preprocesamiento(self):
        img = np.random.default_rng(8).integers(0,256,(90,90),dtype=np.uint8)
        mat, vector = preparar_rostro_desde_gris(img,(80,80))
        self.assertEqual(mat.shape,(80,80))
        self.assertEqual(vector.shape,(6400,))
        self.assertTrue(np.array_equal(mat.ravel(), vector))


if __name__ == '__main__':
    unittest.main()
