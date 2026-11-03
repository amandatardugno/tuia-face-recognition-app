# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── models/
│       │   ├── custom_cnn.py    # CNN custom (Modelo A) y exportación a ONNX
│       │   └── augmentation.py  # Augmentation para entrenar la CNN
│       ├── services/
│       │   ├── face_service.py  # Detección, alineación y embeddings (implementado)
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

1. `_load_detector`: carga el detector SCRFD (`DETECTOR_MODEL_NAME=det_10g.onnx`) con `insightface.model_zoo`, una sola vez, configurado con `DET_INPUT_SIZE` y `DET_SCORE_THRESHOLD`. Se sobrescribe porque el detector viene de una librería; se usa `model_zoo` en lugar de `FaceAnalysis` para que el modelo se tome de `models/` y se configure por `.env`.
2. `detect_faces`: detecta con SCRFD y devuelve las cajas `(x1, y1, x2, y2)` en píxeles enteros, recortadas con `_clip_xyxy` y filtradas por `MIN_FACE_SIZE`. Sin caras, devuelve `[]`.
3. `align_face`: recorta la caja con un margen de `ALIGN_MARGIN`, vuelve a correr SCRFD sobre el recorte para obtener los 5 puntos de la cara (si hay varias, la de mayor IoU con la caja) y alinea con `insightface.utils.face_align.norm_crop` a `FACE_SIZE`. Los keypoints se devuelven relativos a la esquina de la caja. Si no se encuentran puntos, devuelve el recorte redimensionado con `keypoints=None` y deja un warning.
4. `extract_embedding_from_face`: preprocesa con `FaceService.preprocess_face` (BGR → RGB, `(x − 127,5) / 127,5`, CHW), corre el modelo (`onnxruntime` para `.onnx`, `torch` para `.pth`) y devuelve el embedding normalizado L2 como `list[float]` de `EMBEDDING_DIM` valores.

Código agregado para las notebooks: `src/lib/models/custom_cnn.py` (la CNN custom, sus cabezas de entrenamiento y la exportación a ONNX) y `src/lib/models/augmentation.py` (la augmentation del entrenamiento).

## Modelo y fine-tuning

**Modelos (link público de solo lectura):** https://drive.google.com/drive/folders/1ruWQ6GBKVxtDvGqGszbYETTPRP6PxotC?usp=sharing. Copiar los `.onnx` a `models/`; el `LEEME.txt` de la carpeta tiene la fuente, la licencia y los sha256 de cada archivo.

**Modelo elegido: ArcFace preentrenado** (`w600k_r50.onnx`, paquete `buffalo_l` de InsightFace): una ResNet-50 entrenada con la pérdida ArcFace sobre WebFace600K, con embedding de 512 dimensiones, usada sin reentrenar, junto con el detector SCRFD (`det_10g.onnx`). Licencia de InsightFace: uso no comercial y de investigación.

**Modelo comparado: CNN custom** (`cnn_custom.onnx`, `src/lib/models/custom_cnn.py`): 4 bloques convolucionales (dos convoluciones 3×3 con BatchNorm y ReLU, y max pooling), global average pooling, Dropout 0,2 y una capa lineal a 512 con BatchNorm; 1,3 millones de parámetros. Se entrenó desde cero sobre 90 personas de Pins, con dos cabezas: softmax y ArcFace (s = 64, m = 0,5). Hiperparámetros: SGD con momentum 0,9, lr 0,1 con cosine annealing, weight decay 5e-4, lotes de 128, hasta 40 épocas con early stopping (paciencia 5) y el mejor checkpoint según el acierto en validación; augmentation solo en train. Se exportó la de softmax, por mayor AUC en la validación de LFW.

**Resultados** (personas que la CNN no vio; detalle en `2-train.ipynb`):

| Métrica (`lfw_test`) | ArcFace preentrenado | CNN custom (softmax) |
|---|---|---|
| AUC | 0,9998 | 0,948 |
| EER | 0,0010 | 0,123 |
| TAR @ FAR = 1e-3 | 0,999 | 0,301 |
| Top-1 / Top-5 | 1,000 / 1,000 | 0,569 / 0,789 |
| Precisión / recall / F1 macro | 0,564 / 0,999 / 0,681 | 0,416 / 0,477 / 0,379 |
| Exactitud en los pares oficiales de LFW | 0,9985 | 0,879 |
| Tamaño / parámetros | 174 MB / 43,6 M | 5,2 MB / 1,3 M |
| Tiempo por cara en CPU | 65 ms | 3,8 ms |

La CNN rinde mucho menos por los datos (90 personas contra unas 600.000), la capacidad (30 veces menos parámetros) y la pérdida (la cabeza ArcFace quedó subentrenada). Sus falsos positivos muestran que aprendió rasgos generales (edad, origen, anteojos) más que identidades, y sus falsos negativos son fotos con gorras o anteojos de sol. Los peores falsos negativos de ArcFace resultaron ser errores de etiqueta de LFW.

**Umbral:** `SIMILARITY_THRESHOLD=0.201` (coseno), calibrado en la validación de LFW como el que deja pasar 1 de cada 1.000 pares de personas distintas (FAR = 1e-3), con un FRR de 0,0001. El EER, que se usó como control, da 0,245. En identificación con galerías grandes, ese umbral acepta más desconocidos (por la cantidad de comparaciones); con las pocas personas que se registran en la app, el efecto es chico.

## Dataset

Dos datasets públicos, combinados (detalle en `1-pre-processing.ipynb`):

- **Pins Face Recognition** (Kaggle, fotos de Pinterest; la licencia no se pudo verificar): 105 celebridades y 17.534 fotos (de 86 a 237 por persona, casi balanceado). Se usa para entrenar la CNN: 90 personas sorteadas para train y validación (85/15 por persona) y 15 como test del mismo dominio.
- **LFW** (Huang et al., 2007, uso académico): 5.749 personas y 13.233 fotos de prensa, muy desbalanceado (el 71% tiene una sola foto). Se usa para calibrar el umbral y evaluar: dividido por persona entre validación y test, más los pares oficiales. Las 15 personas que también están en Pins se sacaron de LFW.
- **Variaciones:** Pins tiene recortes ajustados y más variación de pose, maquillaje y edición, con un 7,2% de fotos en gris; LFW tiene la cara más chica y centrada, con fondo y a veces otras caras.
- **Preprocesamiento:** se agrega un borde reflejado del 25% antes de detectar (sin él, en Pins no se detecta el 90% de las caras) y se usan la detección y la alineación de la app.
- **Reglas de calidad:** se descartan las fotos sin cara (60), sin puntos para alinear (5), los duplicados de la misma persona (632, coseno > 0,95), las fotos con dos nombres (12), las caras que se parecen más a otra persona (21) y las personas de LFW cargadas con dos nombres (56 personas). No se descarta por tamaño, foco ni exposición: los datos mostraron que esas caras se reconocen, y un umbral de brillo habría descartado sobre todo a personas de piel oscura.
- **Resultado:** 16.851 caras de 105 personas de Pins y 13.015 caras de 5.669 personas de LFW, sin fuga entre conjuntos.

## Notebooks

Las dos notebooks están ejecutadas. Para volver a correrlas hace falta, además de `requirements.txt`, `pip install ipykernel` (y `nbformat`/`nbclient` para ejecutarlas por consola). Los datos no se versionan y van en `data/`: Pins se descarga de Kaggle (dataset "Pins Face Recognition") y se descomprime como `data/105_classes_pins_dataset/`; LFW se baja del espejo que usa `scikit-learn` (`https://ndownloader.figshare.com/files/5976018`, que se extrae como `data/lfw/`, y los pares oficiales de `https://ndownloader.figshare.com/files/5976006` en `data/raw/lfw_pairs/pairs.txt`). `1-pre-processing.ipynb` genera en `data/intermedios/` los splits que usa `2-train.ipynb`. La detección de todas las fotos y el entrenamiento se hicieron en GPU (RTX 3060); en CPU funcionan, pero tardan horas.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
