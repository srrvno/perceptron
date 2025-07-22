# PROMPT

en el texto que me has dado antes, la seccion de 'Bucle de entrenamiento', me ha dado una idea para comprenderr aun mejor como funciona esto, veo que has dividido el proceso en 4 pasos:

    forward(X)        -> p
    loss(p, y)        -> L
    backward(p, y)    -> grads
    update(params, grads, η)

, esto me ha hecho ver la subdivision del mismo y de esta manera me queda mas claro, podrias explicarme el proceso separado por estos pasos, es decir los calculos y proceso que entran en cada paso, para hacer un flujo lineal que creo que es lo que me falla ver, te lo digo yo vervalmente y me corriges si me equivoco:

Para cada instancia del set:

1. Aplicamos el fordwar, es decir en base a las entradas del dataset se propagan hacia la salida haciendose los calculos correspondientes, a cada neurona llegan sus entradas, se calcula la z, luego esta z pasa por la fx de activacion y hay una salida que va a la siguiente neurona si la hay. hasta la capa de salida, lo que produce una salida one, hot, un vector en este caso de dos digitos.

2. Se calcula la perdida con la funcion que se elija, por ejemplo el error cuadratico medio

3. Se calcula el gradiente de la capa de salida
