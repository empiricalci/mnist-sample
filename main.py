import mnist_loader
import network

epochs = 10
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
ev_cost, ev_acc, tr_cost, training_accuracy = net.SGD(
    training_data, epochs, 10, 3.0,
    evaluation_data=test_data,
    monitor_training_accuracy=True
)

# Save model
net.save('/workspace/network.json')

# Save overall
import json
results = json.dumps({
    'metric': 'Accuracy',
    'value': float(training_accuracy[epochs - 1])/50000.0
})

f = open('/workspace/overall.json','w')
f.write(results)
f.close()

# Save training accuracy
table = [['Epoch', 'Accuracy']]
for row, acc in enumerate(training_accuracy):
    table.append([row, float(acc/5000.0)])
t = open('/workspace/accuracy.json', 'w')
t.write(json.dumps(table))
t.close()
