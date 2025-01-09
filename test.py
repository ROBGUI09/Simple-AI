from utils import NeuralNetwork
import random

def on_epoch(epoch, loss):
      if epoch % 1000 == 0:
          print(f"Эпоха {epoch}, ошибка: {loss}")

if __name__ == "__main__":
    layers_config = (1, 5, 1) # 1 вход, 5 нейронов скрытого слоя, 1 выход

    nn = NeuralNetwork(layers_config)

    training_data = []
    for i in range(1000):
        x = random.random()
        y = 1-x
        training_data.append([x, y])

    learning_rate = 0.1
    epochs = 10000

    nn.train(training_data, learning_rate, epochs, callback = on_epoch)

    test_x = 0.7
    test_y = 1 - test_x
    output_test = nn.predict(test_x)
    print(f"\nТест: x = {test_x}, Предсказание: {output_test[0]}, Ожидание: {test_y}")

    nn.save_weights("weights.pkl")
    nn2 = NeuralNetwork(layers_config)
    nn2.load_weights("weights.pkl")

    output_test_loaded = nn2.predict(test_x)
    print(f"\nТест после загрузки: x = {test_x}, Предсказание: {output_test_loaded[0]}, Ожидание: {test_y}")
