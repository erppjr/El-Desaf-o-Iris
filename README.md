# El-Desaf-o-Iris
# Tu Misión: Clasificar y Agrupar Especies de Flores

## Introducción

¡Bienvenido/a al **Desafío Iris**! En esta misión, te pondrás en la piel de un/a científico/a de datos botánico. Tu objetivo es utilizar el famoso dataset **Iris** para clasificar especies de flores. Pero aquí está el reto: lo harás de dos maneras completamente diferentes para descubrir el poder y las limitaciones de cada enfoque.

Primero, usarás el **aprendizaje supervisado** para entrenar un modelo que aprenda a partir de etiquetas existentes. Luego, cambiarás de rol y utilizarás el **aprendizaje no supervisado** para ver si un algoritmo puede descubrir las especies por sí mismo, sin ninguna pista. Esta actividad es la aplicación directa de los conceptos de la **Unidad 2**.

## Objetivos de la Misión 🎯

- Implementar un modelo de **clasificación (supervisado)** y un modelo de **clustering (no supervisado)** usando **Python** y **Scikit-learn**.
- Aplicar técnicas de evaluación para ambos tipos de modelos.
- Visualizar y comparar los resultados para entender intuitivamente cómo funciona cada enfoque.
- Analizar críticamente las **diferencias fundamentales**, ventajas y limitaciones entre el aprendizaje supervisado y no supervisado, conectando la práctica con la teoría.

## Herramientas Requeridas 💻

- **Lenguaje**: Python 3.x
- **Librerías**: 
  - Scikit-learn (para los modelos y el dataset)
  - NumPy, Pandas (para manejo de datos)
  - Matplotlib y Seaborn (para visualizaciones)
- **Entorno de Desarrollo**: Se recomienda Google Colab o un Jupyter Notebook local.

## Fases de la Misión

### Fase 1: Exploración del Jardín Botánico (Análisis de Datos)
Antes de aplicar cualquier algoritmo, debemos conocer nuestro material de trabajo: los datos de las flores.

1. **Carga del Dataset**: Scikit-learn incluye el dataset Iris. Cárgalo usando `sklearn.datasets.load_iris`.
2. **Conversión a DataFrame**: Para facilitar su manipulación, convierte los datos y las etiquetas en un **DataFrame** de Pandas.
3. **Análisis Exploratorio**:
   - Usa `.head()`, `.info()` y `.describe()` para obtener una visión general de los datos.
   - **Visualización Clave**: Utiliza `seaborn.pairplot(dataset, hue='species')` para crear una gráfica que muestre la relación entre todas las características, coloreando los puntos por su especie real.
4. **Pregunta de Análisis 1**: Observando el *pairplot*, ¿parece que las especies forman grupos visualmente separables? ¿Hay alguna especie que sea más fácil de distinguir que las otras? Esta intuición visual es lo que intentaremos que nuestros algoritmos aprendan y descubran.

---

### PARTE A: APRENDIZAJE SUPERVISADO (El Taxónomo Experto)
En esta parte, actuarás como un experto que ya conoce las especies y le enseñará a la máquina a reconocerlas.

#### Fase 2: Entrenamiento del Clasificador
1. **Preparación de Datos**:
   - Separa tus datos en **características (X)** y la **etiqueta objetivo (y, la especie)**.
   - Divide el dataset en un **conjunto de entrenamiento (80%)** y un **conjunto de prueba (20%)** usando `train_test_split`. Esto es crucial para evaluar si nuestro modelo realmente ha aprendido o solo ha memorizado.
2. **Construcción y Entrenamiento del Modelo**:
   - Vamos a usar un **Árbol de Decisión**, un modelo muy interpretable que se explica en el **Tema 2**.
   - Crea una instancia de `DecisionTreeClassifier` y entrénala usando el método `.fit()` con tus datos de entrenamiento (`X_train`, `y_train`).
3. **Evaluación del Modelo**:
   - Realiza predicciones sobre el conjunto de prueba (`X_test`) usando `.predict()`.
   - Calcula la **precisión (accuracy)** del modelo comparando las predicciones con las etiquetas reales (`y_test`).
4. **Pregunta de Análisis 2**: Presenta la precisión obtenida. ¿Consideras que el modelo es bueno? ¿Qué le permitió al algoritmo aprender a clasificar las flores con esta precisión? (Pista: ¿Qué información clave le dimos durante el entrenamiento?)

---

### PARTE B: APRENDIZAJE NO SUPERVISADO (El Explorador Ciego)
Ahora, imagina que no tienes idea de qué especies existen. Tu misión es agrupar las flores por similitud y ver si descubres las categorías naturales por ti mismo.

#### Fase 3: Descubrimiento de Grupos con K-Means
1. **Determinando el Número de Grupos (K)**:
   - El algoritmo **K-Means** necesita que le digamos cuántos grupos (**k**) buscar. Para encontrar el número óptimo, utiliza el **"Método del Codo" (Elbow Method)**, como se describe en el **Tema 2**.
   - Calcula la **inercia** (suma de las distancias al cuadrado) para un rango de valores de **k** (por ejemplo, de 1 a 10).
   - Grafica los resultados. El **"codo"** en la gráfica te indicará el valor óptimo de **k**.
   - **Pregunta de Análisis 3**: ¿Qué valor de **k** sugiere el Método del Codo? ¿Te parece una coincidencia interesante, considerando lo que ya sabes sobre este dataset?
2. **Aplicación de K-Means**:
   - Ejecuta el algoritmo `KMeans` sobre todas las características (`X`), usando el valor de **k** que encontraste.
   - Obtén las **etiquetas de los clústeres** que el algoritmo asignó a cada flor.
3. **Visualización de los Clústeres**:
   - Crea un nuevo **gráfico de dispersión** (por ejemplo, *petal length* vs *petal width*).
   - Esta vez, colorea los puntos según la **etiqueta del clúster** asignada por K-Means.
   - Añade los **centroides** (los centros de cada clúster) al gráfico para visualizarlos.

#### Fase 4: La Gran Comparación y Conclusión
Este es el momento de la verdad. Comparemos lo que el explorador "ciego" descubrió con la realidad.

1. **Comparación de Resultados**:
   - Crea una **tabla de contingencia** (puedes usar `pd.crosstab`) que compare las **etiquetas reales** de las especies con las **etiquetas de los clústeres** encontradas por K-Means.
2. **Análisis Final**:
   - **Pregunta de Análisis 4**: Observando la tabla de contingencia, ¿qué tan bien se corresponden los clústeres con las especies reales? ¿Logró el aprendizaje no supervisado descubrir la estructura oculta de los datos?
   - **Pregunta de Análisis 5**: Basándote en tu experiencia en esta actividad y en la teoría de la **Unidad 2**, resume la **diferencia fundamental** entre el enfoque supervisado y el no supervisado. ¿Cuándo usarías uno sobre el otro en un problema del mundo real?

---

## Entregables

1. **El Notebook (.ipynb)**:
   - Debe contener todo tu **código**, ejecutado y con todas las **salidas visibles** (tablas, gráficos, etc.).
   - El código debe estar **comentado** para explicar tus decisiones.
   - Incluye las respuestas a las **cinco Preguntas de Análisis** en celdas de texto (Markdown).

2. **Informe Técnico (PDF, 2-3 páginas)**:
   - **Introducción**: Describe brevemente el problema y los dos enfoques de *Machine Learning* que se explorarán.
   - **Metodología y Resultados - Aprendizaje Supervisado**: Explica el modelo de Árbol de Decisión, presenta su precisión y discute su rendimiento.
   - **Metodología y Resultados - Aprendizaje No Supervisado**: Explica cómo determinaste **k** con el Método del Codo e incluye la visualización de los clústeres.
   - **Discusión Comparativa y Conclusión**: Esta es la sección crucial. Compara directamente los resultados de ambos métodos. Discute las **ventajas y desventajas** de cada uno basándote en tu experiencia. Concluye explicando qué aprendiste sobre la **naturaleza de los datos** y la **aplicabilidad de cada paradigma de Machine Learning**.
