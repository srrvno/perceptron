# Perceptrón

## 1. Perceptrón simple

### 1.1 Origen e idea básica

* Una única neurona artificial para aprendizaje supervisado, propuesta originalmente por **Frank Rosenblatt (1957)** como clasificador binario.
* Se comporta como **clasificador lineal**: una suma ponderada decide entre dos clases.

### 1.2 Estructura matemática

Cada neurona está definida por: entradas $x_i$, pesos $w_i$, un **umbral** $U$ y una función de activación $f(\cdot)$.

<div style="text-align:center;">
  <img src="docs/images/image.png" alt="perceptron" width="400" />
</div>

$$
a = \sum_{i=1}^{n} w_i\,x_i - U
$$

La salida continua (probabilidad) se obtiene con la **sigmoide**:

$$
y = f(a) = \sigma(a) = \frac{1}{1+e^{-a}}
$$


### 1.3 Algoritmo de aprendizaje  (sigmoide + log‑loss)

1. **Inicializar** pesos y umbral.

   * $w_i\sim[-0.5,0.5]$
   * $U\sim[0,1]$
2. Para cada ejemplo $(\mathbf{x},s)$ (etiqueta $s\in{0,1}$):

   1. Calcular $a=\sum w_i x_i-U$ y $y=\sigma(a)$.
   2. **Error suave**: $\delta = y-s$.
   3. **Actualizar** parámetros (descenso de gradiente de la log‑loss):

$$
\begin{aligned}
 w_i &\leftarrow w_i - \alpha\delta x_i \\[4pt]
 U   &\leftarrow U   + \alpha\,\delta
\end{aligned}
$$

3. Repetir durante varias épocas hasta que la pérdida media se estabilice.

### 1.4 Inferencia

1. Calcular $a$ y $y=\sigma(a)$ sobre datos de test.
2. **Umbralizar** (decisión):
   $(y\ge0.5)\Rightarrow1$; 

   $(y<0.5)\Rightarrow0$.

---

## 2. Perceptrón multicapa (MLP)

### 2.1 Motivación

Capas ocultas permiten aprender fronteras **no lineales**.

### 2.2 Arquitectura

<img src="docs/images/image-5.png" alt="mlp" width="460" />

### 2.3 Aprendizaje por retro‑propagación

Para cada peso $w\_{ij}$:  

$w_{ij} \leftarrow w_{ij} - \alpha\,\frac{\partial E}{\partial w_{ij}}$

Requiere activaciones derivables (sigmoide, ReLU, tanh…).

---

## 3. Funciones de activación

| Tipo          | Fórmula                                   | Gráfica                                                         |
| ------------- | ----------------------------------------- | --------------------------------------------------------------- |
| **Lineal**    | $f(a)=a$                                | —                                                               |
| **Sigmoide**  | $\displaystyle f(a)=\frac{1}{1+e^{-a}}$ | <img src="docs/images/image-2.png" alt="sigmoid" width="190" /> |
| **Gaussiana** | selectiva a valores intermedios           | <img src="docs/images/image-1.png" alt="gauss" width="190" />   |
| **ReLU**      | $f(a)=\max(0,a)$                        | —                                                               |

---

## 4. Resumen rápido de fórmulas clave

* **Agregación:** $a=\sum w_i x_i - U$
* **Salida continua:** $y=\sigma(a)$
* **Error suave:** $\delta = y - s$
* **Actualización sigmoide/log‑loss:**
  $w\_i \leftarrow w\_i - \alpha \delta x_i$,  
  $U \leftarrow U + \alpha\delta$
* **Gradiente MLP:** $w \leftarrow w - \alpha\nabla E$
