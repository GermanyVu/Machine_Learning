import feedforward_network as ff
import data_processing as dp


epochs = 40
mini_batch_size = 10
data_name = "mnist_784"
training_data, test_data = dp.fetch_data(data_name)

ffnn = ff.FFNN()
ffnn.add_layer(784)
ffnn.add_layer(100)
ffnn.add_layer(10)
ffnn.make_weights()

print(ffnn)
ffnn.stochastic_gradient(training_data, epochs, mini_batch_size, test_data)
