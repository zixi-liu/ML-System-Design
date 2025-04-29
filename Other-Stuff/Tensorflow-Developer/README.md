
### Data Representations

```
model = tf.teras.Sequential([
  tf.keras.Input(shape=(1,)),
  tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sdg', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0], dtype = float)

model.fit(xs, ys, epochs = 500)

model.predict([10.0])
```
