package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type SigmoidalNeuron struct {
	inputSignal  float64
	outputSignal float64
}

func NewSigmoidalNeuron() *SigmoidalNeuron {
	return &SigmoidalNeuron{0, 0}
}

func (n *SigmoidalNeuron) calcActInput(input float64) float64 {
	n.inputSignal = input
	n.outputSignal = n.actInput(input)
	return n.outputSignal
}

func (n *SigmoidalNeuron) calcActHidden(input float64) float64 {
	n.inputSignal = input
	n.outputSignal = n.actHidden(input)
	return n.outputSignal
}

func (n *SigmoidalNeuron) calcActOutput(input float64) float64 {
	n.inputSignal = input
	n.outputSignal = n.actOutput(input)
	return n.outputSignal
}

func (n *SigmoidalNeuron) Through(input float64) float64 {
	n.outputSignal = input
	n.inputSignal = input
	return input
}

func (n *SigmoidalNeuron) actInput(x float64) float64 {
	// return (2.0 / (1.0 + math.Exp(-x))) - 1.0
	return (1.0 / (1.0 + math.Exp(-x))) // org
	// return math.Tanh(x)
}

func (n *SigmoidalNeuron) actHidden(x float64) float64 {
	// return (2.0 / (1.0 + math.Exp(-x))) - 1.0
	// return math.Max(0.0, x)
	return (1.0 / (1.0 + math.Exp(-x))) // org
	// return math.Tanh(x)
}

func (n *SigmoidalNeuron) actOutput(x float64) float64 {
	// return (2.0 / (1.0 + math.Exp(-x))) - 1.0
	return math.Tanh(x)
	// return (1.0 / (1.0 + math.Exp(-x))) - 0.5 // org
}

type DimError struct {
	What string
}

func (e *DimError) Error() string {
	return fmt.Sprintf("at %s", e.What)
}

type NeuralNetwork struct {
	inputDim, hiddenDim, outputDim          int
	inputNeuron, hiddenNeuron, outputNeuron []SigmoidalNeuron
	wih, who                                [][]float64
}

func NewRandom2dSlice(nInputs int, nOutputs int) [][]float64 {

	lower, upper := -(math.Sqrt(6.0) / math.Sqrt(float64(nInputs+nOutputs))), (math.Sqrt(6.0) / math.Sqrt(float64(nInputs+nOutputs))) // Norm Xavier
	// lower, upper := -(1.0 / math.Sqrt(float64(nInputs))), (1.0 / math.Sqrt(float64(nInputs))) // Xavier
	// lower := math.Sqrt(float64(2) / float64(nInputs)) // He
	// lower, upper := -(math.Sqrt(3.0 / float64(nInputs))), (math.Sqrt(3.0 / float64(nInputs))) // Xavier

	s := make([][]float64, nInputs)
	for i := 0; i < nInputs; i++ {
		s[i] = make([]float64, nOutputs)
		for j := 0; j < nOutputs; j++ {
			s[i][j] = lower + rand.Float64()*(upper-lower)
			// s[i][j] = rand.Float64() * (lower)
		}
	}
	return s
}

func RandomWeightsHidden(nInputs int, nOutputs int) [][]float64 {

	std := math.Sqrt(2.0 / float64(nInputs))

	s := make([][]float64, nInputs)
	for i := 0; i < nInputs; i++ {
		s[i] = make([]float64, nOutputs)
		for j := 0; j < nOutputs; j++ {
			s[i][j] = rand.Float64() * (std)
		}
	}
	return s
}

func NewNeuralNetwork(inputNum int, hiddenNum int, outputNum int) *NeuralNetwork {
	wih := RandomWeightsHidden(inputNum+1, hiddenNum)
	// who := NewRandom2dSlice(hiddenNum+1, outputNum)
	who := RandomWeightsHidden(hiddenNum+1, outputNum)
	return &NeuralNetwork{inputNum, hiddenNum, outputNum,
		make([]SigmoidalNeuron, inputNum+1),
		make([]SigmoidalNeuron, hiddenNum+1),
		make([]SigmoidalNeuron, outputNum),
		wih, who}
}

// calc Calcula la salida de la red para un vector de entrada y pesos establecidos
func (nn *NeuralNetwork) calc(inputVec []float64) ([]float64, error) {
	if len(inputVec) != nn.inputDim {
		return nil, &DimError{"input error"}
	}

	// Calcula la salida de cada neurona de la capa de entrada
	for i := 0; i < nn.inputDim; i++ {
		nn.inputNeuron[i].calcActInput(inputVec[i])
	}
	nn.inputNeuron[nn.inputDim].Through(1.0) // --Aux

	// Calcula la salida de cada neurona de la capa oculta
	var sum float64
	for h := 0; h < nn.hiddenDim; h++ {
		sum = 0.0
		for i := 0; i < nn.inputDim+1; i++ {
			sum += nn.inputNeuron[i].outputSignal * nn.wih[i][h]
		}
		nn.hiddenNeuron[h].calcActHidden(sum)
	}
	nn.hiddenNeuron[nn.hiddenDim].Through(1.0) // -- Aux

	// Calcula la salida de cada neurona de la capa de salida
	for o := 0; o < nn.outputDim; o++ {
		sum = 0.0
		for h := 0; h < nn.hiddenDim+1; h++ {
			sum += nn.hiddenNeuron[h].outputSignal * nn.who[h][o]
		}
		nn.outputNeuron[o].calcActOutput(sum)
	}
	outputVec := make([]float64, nn.outputDim)
	for j := 0; j < nn.outputDim; j++ {
		outputVec[j] = nn.outputNeuron[j].outputSignal
	}
	return outputVec, nil
}

// Algoritmo back propagation con opción de implementar weight decay
func (nn *NeuralNetwork) backpropagation(inputVec []float64, targetVec []float64, learningConst float64, tolerance float64) error {
	if nn.inputDim != len(inputVec) {
		return &DimError{"inputVec dim error"}
	}
	if nn.outputDim != len(targetVec) {
		return &DimError{"targetVec dim error"}
	}

	// Paso forward (se calculan las salidas)
	outputVec, _ := nn.calc(inputVec)

	convergeState := true
	for k := 0; k < nn.outputDim; k++ {
		diff := targetVec[k] - outputVec[k]
		if diff > tolerance || diff < -tolerance {
			convergeState = false
		}
	}
	if convergeState == true {
		return nil
	}

	//backpropagation
	delta_o := make([]float64, nn.outputDim)
	delta_h := make([]float64, nn.hiddenDim)
	tmpPreWho := make([][]float64, nn.hiddenDim+1)
	for i := 0; i < nn.hiddenDim+1; i++ {
		tmpPreWho[i] = make([]float64, nn.outputDim+1)
	}

	for k := 0; k < nn.outputDim; k++ {

		// dE_total/d_w[k] = delta_o[k] * nn.hiddenNeuron[j].outputSignal
		delta_o[k] = (targetVec[k] - outputVec[k]) * derivativeTh(outputVec[k])

		for j := 0; j < nn.hiddenDim+1; j++ {
			tmpPreWho[j][k] = nn.who[j][k]
			// delta_o[k] = - dE_total/do[k] * do/d_net[k] = dE_total/d_net[k]
			// nn.who[j][k] += learningConst * delta_o[k] * nn.hiddenNeuron[j].outputSignal
			nn.who[j][k] = nn.who[j][k] + learningConst*delta_o[k]*nn.hiddenNeuron[j].outputSignal - (0.001 * learningConst * nn.who[j][k])
		}
	}

	for j := 0; j < nn.hiddenDim; j++ {
		delta_h[j] = 0.0
		for k := 0; k < nn.outputDim; k++ {
			delta_h[j] += delta_o[k] * tmpPreWho[j][k]
		}
		delta_h[j] *= nn.hiddenNeuron[j].outputSignal * (1.0 - nn.hiddenNeuron[j].outputSignal)
		for i := 0; i < nn.inputDim+1; i++ {
			// nn.wih[i][j] += learningConst * delta_h[j] * nn.inputNeuron[i].outputSignal
			nn.wih[i][j] = nn.wih[i][j] + learningConst*delta_h[j]*nn.inputNeuron[i].outputSignal - (0.001 * learningConst * nn.wih[i][j])
		}
	}
	return nil
}

func derivativeSig2(x float64) float64 {
	return 2.0 * math.Exp(-x) * (1.0 + math.Exp(-x)) * (1.0 + math.Exp(-x))
}

func derivativeTh(x float64) float64 {
	return 1.0 - math.Pow(math.Tanh(-x), 2.0)
}

func main() {

	rand.Seed(time.Now().UnixNano())

	// Genera la red neuronal:
	// - 20 neuronas en la capa de entrada
	// - 50 neuronas en la capa oculta (una sola capa oculta)
	// - 1 neurona en la capa de salida
	nn := NewNeuralNetwork(20, 50, 1)
	x_, _ := readTrainingData()

	// Se dividela data de entrenamiento y validación
	xTrain := x_[:700]
	xTest := x_[700:1000]
	fmt.Println(len(x_), len(x_[0]), len(xTrain), len(xTest))

	ok, mal := 200, 250
	// best := 300
	for mal > 48 {
		// En cada réplica se inicializan los pesos
		nn = NewNeuralNetwork(20, 50, 1)
		for epoch := 0; epoch < 50000; epoch++ {
			rand.Shuffle(len(xTrain), func(i, j int) { xTrain[i], xTrain[j] = xTrain[j], xTrain[i] })
			for _, x := range xTrain {
				nn.backpropagation(x[:20], []float64{x[20]}, 0.01, 0.000001)
			}

			if epoch%100 == 0 {
				ok, mal = 0, 0
				for _, x_ := range xTrain {
					outVec, err := nn.calc(x_[:20])
					if err != nil {
						fmt.Println(err.Error())
						return
					}

					if (outVec[0] < 0 && x_[20] < 0) || (outVec[0] > 0 && x_[20] > 0) {
						ok++
					} else {
						mal++
					}
				}
				// fmt.Println("Training", "Ok", ok, "Wrong", mal, (float64(mal) / float64((ok + mal))))

				if mal < 35 {
					fmt.Println("Good training", "Ok", ok, "Wrong", mal, (float64(mal) / float64((ok + mal))), "epoch", epoch)
					break
				}
			}

		}
		ok, mal = 0, 0
		for _, x_ := range xTest {
			outVec, err := nn.calc(x_[:20])
			if err != nil {
				fmt.Println(err.Error())
				return
			}

			if (outVec[0] < 0 && x_[20] < 0) || (outVec[0] > 0 && x_[20] > 0) {
				ok++
			} else {
				mal++
			}
		}
		fmt.Println("Test: Ok", ok, "Wrong", mal, float64(mal)/float64((ok+mal)))

		// if mal < best {

		// }
	}

	name := "decay_sig_sig_tanh_Xavier_001" + strconv.FormatInt(int64(mal), 10) + ".csv"
	csvFile, err := os.Create(name)

	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}

	csvwriter := csv.NewWriter(csvFile)

	for _, neuron := range nn.who {
		var weights_ []string
		for _, weight := range neuron {
			weights_ = append(weights_, strconv.FormatFloat(weight, 'f', -1, 64))
		}
		_ = csvwriter.Write(weights_)
	}

	for _, neuron := range nn.wih {
		var weights_ []string
		for _, weight := range neuron {
			weights_ = append(weights_, strconv.FormatFloat(weight, 'f', -1, 64))
		}
		_ = csvwriter.Write(weights_)
	}

	csvwriter.Flush()
	csvFile.Close()

	// Test

	dataTest, _ := readTestData()
	name = "labels.csv"
	csvFile_, err := os.Create(name)
	csvwriter_ := csv.NewWriter(csvFile_)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	for _, x_ := range dataTest {
		outVec, err := nn.calc(x_[:20])
		if err != nil {
			fmt.Println(err.Error())
			return
		}
		if outVec[0] > 0.0 {
			outVec[0] = 1
		} else {
			outVec[0] = -1
		}
		_ = csvwriter_.Write([]string{strconv.FormatFloat(outVec[0], 'f', -1, 64)})
	}

	csvwriter_.Flush()
	csvFile_.Close()

	fmt.Println("Fin")
	fmt.Println("Test\nOk", ok, "Wrong", mal)

}

func readTrainingData() (x_ [][]float64, y_ []float64) {
	//read data
	irisMatrix := [][]string{}
	iris, err := os.Open("data.csv")
	if err != nil {
		panic(err)
	}
	defer iris.Close()

	reader := csv.NewReader(iris)
	reader.Comma = ','
	reader.LazyQuotes = true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		irisMatrix = append(irisMatrix, record)
	}

	//separate data into explaining and explained variables
	X := [][]float64{}
	Y := []float64{}

	for _, data := range irisMatrix {
		//convert str slice to float slice
		temp := []float64{}
		for _, i := range data[:len(data)-1] {
			parsedValue, err := strconv.ParseFloat(i, 64)
			if err != nil {
				panic(err)
			}
			temp = append(temp, parsedValue)
		}

		temp = append([]float64{}, temp...)
		if data[len(data)-1] == "-1" {
			temp = append(temp, -1.0)
			Y = append(Y, -1.0)
		} else {
			temp = append(temp, 1.0)
			Y = append(Y, 1.0)
		}
		X = append(X, temp)
	}

	return X, Y
}

func readTestData() (x_ [][]float64, y_ []float64) {
	//read data
	irisMatrix := [][]string{}
	iris, err := os.Open("data_test.csv")
	if err != nil {
		panic(err)
	}
	defer iris.Close()

	reader := csv.NewReader(iris)
	reader.Comma = ','
	reader.LazyQuotes = true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		irisMatrix = append(irisMatrix, record)
	}

	//separate data into explaining and explained variables
	X := [][]float64{}
	Y := []float64{}

	for _, data := range irisMatrix {
		//convert str slice to float slice
		temp := []float64{}
		for _, i := range data[:len(data)-1] {
			parsedValue, err := strconv.ParseFloat(i, 64)
			if err != nil {
				panic(err)
			}
			temp = append(temp, parsedValue)
		}

		temp = append([]float64{}, temp...)
		X = append(X, temp)
	}

	return X, Y
}
