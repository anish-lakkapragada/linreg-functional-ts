/**
 * Let's complicate things for no reason. 
 */

 interface LinearReg {
    weight: number; 
    bias: number; 
    learningRate: number;
    epochs: number; 
}

const x : number[] = [1, 2, 3, 4, 5]; 
const y: number[] = x.map(item => item * 2 + 1); 

const prediction =  function (linreg : LinearReg,  x_i: number) : number {
    return linreg.weight * x_i + linreg.bias;
}

const getPredictions = function(linreg: LinearReg, inputs: number[]) : number[] {
    return inputs.map(input => prediction(linreg, input));
}

const mse = function (y_true: number, y_pred : number) : number {
    return Math.pow(y_pred - y_true, 2);
}

const MSE = function(predictions: number[], labels: number[]): number {
    return predictions.map((prediction, i) => mse(labels[i], prediction)).reduce((a, b) => a + b, 0) / predictions.length;
}

const mse_grad = function(prediction: number, label: number) : number {
    return 2 * (prediction - label);
}

const MSE_Grad = function(predictions: number[], labels: number[]) : number[] {
    return predictions.map((prediction, i) => mse_grad(prediction, labels[i])); 
}

const dot = function(a: number[], b: number[]): number {
    return a.map((a_i, i) => a_i * b[i]).reduce((a, b) => a + b, 0); 
}

// dL/dW = 2 * (pred_i - labels_i) * x_i
const weight_grad = function(inputs: number[], predictions: number[], labels: number[]): number {
    return dot(MSE_Grad(predictions, labels), inputs) / predictions.length;
}

// dL/db = 2 * (pred_i - labels_i) 
const bias_grad = function(predictions: number[], labels: number[]) : number {
    return MSE_Grad(predictions, labels).reduce((a, b) => a + b, 0) / predictions.length;
}


const linearReg: LinearReg = {weight: Math.random(), bias: Math.random(), learningRate: 0.01, epochs: 100};

for (let i = 1; i <= linearReg.epochs; i++) {
    const predictions = getPredictions(linearReg, x);
    const dLdW = weight_grad(x, predictions, y);
    const dLdB = bias_grad(predictions, y);
    linearReg.weight -= linearReg.learningRate * dLdW;
    linearReg.bias -= linearReg.learningRate * dLdB;
    
    console.log(`Epoch ${i}: MSE = ${MSE(predictions, y)}`);
}

// Yo this is hella sus it worked on first run
// I guess functional programming is for me? 
