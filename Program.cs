using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

//using NumSharp;
//using static Tensorflow.Binding; // This makes the 'tf' object available






Attempt0Entry entry = new Attempt0Entry();
entry.entry();

int breakpointHere8 = 1;


RunnerEnv runnerEnv = new RunnerEnv();
runnerEnv.run();

int breakpointDebug8 = 1;




//public static class CscgFormulas {
//    /*
//     * /param arrX array of X. which is the sequence of observations as discrete events
//     * 
//     */
//    public static double calcProb(int[] arrX, double[] pZ, double[] pOfZbyZ, double[,] pOfXByZ, int N) {
//        // formula 1 from CSCG paper
//
//        double pInnerA = pZ[0];
//
//        double pInnerB = 1.0;
//        for (int nInner = 0; nInner < N - 1; nInner++) {
//            pInnerB *= pOfZbyZ[nInner]; // pOfZbyZ[nInner + 1, nInner];
//        }
//
//        /*
//        double pInnerC = 1.0;
//        for (int nInner = 0; nInner < N; nInner++) {
//            pInnerC *= pOfXByZ[nInner]; // pOfXByZ[nInner, nInner];
//        }
//        */
//
//        double pInnerC = 1.0;
//        for (int xIdx = 0; xIdx < N; xIdx++) {
//
//            int xVal = arrX[xIdx];
//            int yVal = 0; // TODO: pull our the right yIndex
//
//            pInnerC *= pOfXByZ[xVal, yVal];
//        }
//
//        double p = pInnerA * pInnerB * pInnerC;
//        return p;
//    }
//}

/*

public class Z {
    public void entry() {
        int N = 5;

        double[] pZ = new double[1];
        pZ[0] = 1.0;

        double[] pOfZbyZ = new double[N-1];

        double[,] pOfXByZ = new double[N,N];


        // todo : implement actual optimization

        double prob = CscgFormulas.calcProb(pZ, pOfZbyZ, pOfXByZ, N);
    }

}
*/


// TODO : overhaul with ids of events




// context which contains the matrices
public class Ctx {
    public double[,] transitionMatrix; // matrix with transition probabilities  [idx of current state   ,  transition probability to idx of state]

    public double[,] emissionMatrix; // index by [stateIdx, outputSymbolIdx]

    public int retNumberOfStates() {
        return transitionMatrix.GetLength(0);
    }

    public int retNumberOfSymbols() {
        return emissionMatrix.GetLength(1);
    }

    public int nActions;

    public int calcIdxOfStateAndAction(int state, int action) {
        int res = state + action * retNumberOfStates();

        if (res < 0) {
            int breakpointDebugHere10 = 1;
        }

        return res;
    }
}

// utilities to manipulate context
public class CtxUtils {
    public static Ctx clone(Ctx arg) {
        Ctx cloned = new Ctx();
        cloned.nActions = arg.nActions;
        cloned.transitionMatrix = new double[arg.transitionMatrix.GetLength(0), arg.transitionMatrix.GetLength(1)];
        cloned.emissionMatrix = new double[arg.emissionMatrix.GetLength(0), arg.emissionMatrix.GetLength(1)];

        for (int iy = 0; iy < arg.transitionMatrix.GetLength(0); iy++) {
            for (int ix = 0; ix < arg.transitionMatrix.GetLength(0); ix++) {
                cloned.transitionMatrix[iy, ix] = arg.transitionMatrix[iy, ix];
            }
        }

        for (int iy = 0; iy < arg.emissionMatrix.GetLength(0); iy++) {
            for (int ix = 0; ix < arg.emissionMatrix.GetLength(1); ix++) {
                cloned.emissionMatrix[iy, ix] = arg.emissionMatrix[iy, ix];
            }
        }

        return cloned;
    }

    public static void normalizeTransitionMatrix(Ctx arg) {
        for (int idxSourceState = 0; idxSourceState < arg.transitionMatrix.GetLength(0); idxSourceState++) {

            double probSum = 0.0;

            for (int ix = 0; ix < arg.transitionMatrix.GetLength(1); ix++) {
                probSum += arg.transitionMatrix[idxSourceState, ix];
            }

            for (int ix = 0; ix < arg.transitionMatrix.GetLength(1); ix++) {
                arg.transitionMatrix[idxSourceState, ix] /= probSum;
            }
        }
    }

    public static void normalizeEmissionMatrix(Ctx arg) {
        for (int idxSourceState = 0; idxSourceState < arg.emissionMatrix.GetLength(0); idxSourceState++) {

            double probSum = 0.0;

            for (int ix = 0; ix < arg.emissionMatrix.GetLength(1); ix++) {
                probSum += arg.emissionMatrix[idxSourceState, ix];
            }

            for (int ix = 0; ix < arg.emissionMatrix.GetLength(1); ix++) {
                arg.emissionMatrix[idxSourceState, ix] /= probSum;
            }
        }
    }

    // /param observations  array with observations to be sampled/to learn
    // /param ctx           
    public static List<int> samplePath(List<TupleStateAndAction> observationAndActionList, Ctx ctx, Random rng) {

        int currentState = 0; // index of current state in which we are in currently for sampling


        // trace of states of this sampled trace
        List<int> stateTrace = new List<int>();


        //double pathProbability = 1.0; // overall probability of the sampled path

        // in loop
        foreach (TupleStateAndAction tupleObservedExternalStateAndAction in observationAndActionList) {
            int observationSymbol = tupleObservedExternalStateAndAction.observationState;


            //Console.WriteLine("");
            //Console.WriteLine("");
            //Console.WriteLine($"current state = {currentState}");

            // FIXME : needs to get overhauled . to the new observation#action pair instead the old observation  !?!??!?!?!?!?!??!?!?!??!?!?!

            // * choose next action by random based on transition probabilities
            double sum0 = 0.0;
            for (int idx = 0; idx < ctx.retNumberOfStates(); idx++) {
                sum0 += ctx.transitionMatrix[currentState, idx];
            }

            double chosen = rng.NextDouble() * sum0;
            double accu = 0.0;
            int idxChosen = 0;
            int idx2 = 0; // index of chosen transition
            for (; idx2 < ctx.retNumberOfStates(); idx2++) {
                accu += ctx.transitionMatrix[currentState, idx2];
                if (accu >= chosen) {
                    idxChosen = idx2;
                    break;
                }
            }


            // keep track of the path probability by multiplying by the probability of this choice
            //pathProbability *= (ctxConsidered.transitionMatrix[currentState, idxChosen] / sum0); // division here is to normalize the probabilities

            // keep trace of the state trace
            stateTrace.Add(currentState);

            // * do chosen action and transition to chosen state
            //   (we can't do the action for real here, so we just transition to the state)
            currentState = idxChosen;


            int breakpointHere5 = 1;
        }

        return stateTrace;
    }
}




public static class Utils {
    public static string convToStr(List<int> arg) {
        List<string> arr = new List<string>();

        for (int idx = 0; idx < arg.Count; idx++) {
            arr.Add(arg[idx].ToString());
        }

        return "{" + string.Join(",", arr.ToArray()) + "}";
    }
}

// matrix utils
public static class MatrixUtils {
    public static void fillOne(double[,] m) {
        for (int iy = 0; iy < m.GetLength(0); iy++) {
            for (int ix = 0; ix < m.GetLength(1); ix++) {
                m[iy, ix] = 1.0;
            }
        }
    }
}



// pair of state and done action in that state.
// as described in the paper
public struct TupleStateAndAction {
    public int observationState; // observed external state
    public int action; // done action

    public TupleStateAndAction(int observationState, int action) {
        this.observationState = observationState;
        this.action = action;
    }
}


/*
public class ObservationAndActionList
{
    List<TupleStateAndAction> observationAndActionList = new List<TupleStateAndAction>();
}
*/



public class CscgOptimizer {

    public List<TrainingTrace> trainingTraces = new List<TrainingTrace>();




    public Random rng = new Random(143);


    public int outputSymbolsAlphabetSize = 4;

    public int nActions = 3;
    public int nStates = 5; // number of states

    ///int transitionMatrixSize0 = nStates; // number of states
    ///int transitionMatrixSize1 = 3; // number of states






    // how many times are the paths sampled for a sequence of observations?
    public int nSampledPaths = 5; // 3


    // how many iterations to optimize the model ?
    public int nOptimizationIterations = 3000;



    public int verbosity = 0;



    public Ctx ctxBest = null;

    public CscgOptimizer() {
    }

    public void init() {
        ctxBest = new Ctx();

        ctxBest.nActions = nActions;

        ctxBest.transitionMatrix = new double[nStates, nStates * nActions]; // matrix with transition probabilities  [idx of current state   ,  transition probability to idx of state and action pair]
        ctxBest.emissionMatrix = new double[nStates, outputSymbolsAlphabetSize]; // index by [stateIdx, outputSymbolIdx]



        // init transition matrix
        // init emission matrix
        // Initialize with 1.0 PLUS random noise to break symmetry!
        for (int i = 0; i < nStates; i++) {
            for (int j = 0; j < nStates * nActions; j++) {
                ctxBest.transitionMatrix[i, j] = 1.0 + (rng.NextDouble() * 0.1);
            }
            for (int j = 0; j < outputSymbolsAlphabetSize; j++) {
                ctxBest.emissionMatrix[i, j] = 1.0 + (rng.NextDouble() * 0.1);
            }
        }

        CtxUtils.normalizeTransitionMatrix(ctxBest);
        CtxUtils.normalizeEmissionMatrix(ctxBest);
    }

    // sampling optimization step
    public void simplisticOptimizerStep() {
        
        // sample observations

        ///
        ///List<int> observations = new List<int>(); // array with observations to be sampled/to learn
        ///
        ///observations.Add(0);
        ///observations.Add(2);
        ///

        int selIdx = rng.Next(trainingTraces.Count);
        List<TupleStateAndAction> observationAndActionList = trainingTraces[selIdx].list;





        // optimize

        Ctx ctxConsidered = CtxUtils.clone(ctxBest);



        double expectationOverallSumOfBest = 0.0; // sum of all sampled traces for all considered observations
        double expectationOverallSumOfConsidered = 0.0; // sum of all sampled traces for all considered observations



        for (int samplePathAttempt = 0; samplePathAttempt < nSampledPaths; samplePathAttempt++) {

            // sample path
            List<int> stateTrace = CtxUtils.samplePath(observationAndActionList, ctxConsidered, rng);

            if (verbosity >= 1) {
                Console.WriteLine($"sampled state trace  stateTrace={Utils.convToStr(stateTrace)}");
            }



            // now we need to update the transition matrix and the emission matrix based on the result of the sampling
            {
                // update transition matrix
                {
                    for (int idx = 0; idx < stateTrace.Count - 1; idx++) {
                        int statePrevious = stateTrace[idx];
                        ////int stateNext = stateTrace[idx+1];

                        int idxOfNext = ctxConsidered.calcIdxOfStateAndAction(stateTrace[idx + 1], observationAndActionList[idx].action);  /// stateTrace[idx+1] + observationAndActionList[idx].action * nStates;

                        ctxConsidered.transitionMatrix[statePrevious, idxOfNext] *= 1.05;
                    }
                }



                // push up the probability of the correct transition of the output matrix
                //
                // for every transition
                for (int idxObservation = 0; idxObservation < observationAndActionList.Count; idxObservation++) {
                    int state = stateTrace[idxObservation];
                    int symbolOfObservation = observationAndActionList[idxObservation].observationState;

                    ctxConsidered.emissionMatrix[state, symbolOfObservation] *= 1.05;
                }


                // normalize transition matrix
                CtxUtils.normalizeTransitionMatrix(ctxConsidered);

                // normalize emission matrix
                CtxUtils.normalizeEmissionMatrix(ctxConsidered);

            }


            // debug matrices
            {
                // TODO
            }

            // now we need to calculate expectation and see if expectation got improved (expectation maximization)
            {



                List<TupleStateAndAction> observationAndActionChecked = observationAndActionList;
                List<int> stateTraceChecked = stateTrace;

                double expectationOfTrace = calcExpectation(observationAndActionChecked, stateTraceChecked, ctxConsidered);

                if (verbosity >= 1) {
                    Console.WriteLine($"Expectation of sampled trace of sampled  expectationProb={expectationOfTrace}"); // DBG
                }

                expectationOverallSumOfConsidered += expectationOfTrace;








                observationAndActionChecked = observationAndActionList;
                stateTraceChecked = stateTrace;

                expectationOfTrace = calcExpectation(observationAndActionChecked, stateTraceChecked, ctxBest);

                if (verbosity >= 1) {
                    Console.WriteLine($"Expectation of sampled trace of best  expectationProb={expectationOfTrace}"); // DBG
                }

                expectationOverallSumOfBest += expectationOfTrace;





            }


            // do actual optimization by comparing expectation of bestCtx and then swap it if it is better
            if (expectationOverallSumOfConsidered > expectationOverallSumOfBest) {
                ctxBest = CtxUtils.clone(ctxConsidered);
            }

        }


        int breakpointDebug7 = 1;
    }


    public void viterbiOptimizerStep() {
        // code from LLM Gemini 3.1 Pro (high)

        // 1. Initialize count matrices with a small pseudocount (regularization)
        // The CSCG paper uses pseudocounts (e.g., 1e-2 or 1e-3) to handle unobserved transitions.
        double pseudoCount = 1e-3;
        double[,] transitionCounts = new double[nStates, nStates * nActions];
        double[,] emissionCounts = new double[nStates, outputSymbolsAlphabetSize];

        for (int i = 0; i < nStates; i++) {
            for (int j = 0; j < nStates * nActions; j++) transitionCounts[i, j] = pseudoCount;
            for (int j = 0; j < outputSymbolsAlphabetSize; j++) emissionCounts[i, j] = pseudoCount;
        }

        // Precompute log matrices for numerical stability
        double[,] logTrans = new double[nStates, nStates * nActions];
        double[,] logEmiss = new double[nStates, outputSymbolsAlphabetSize];

        for (int i = 0; i < nStates; i++) {
            for (int j = 0; j < nStates * nActions; j++) {
                logTrans[i, j] = Math.Log(Math.Max(ctxBest.transitionMatrix[i, j], 1e-100));
            }
            for (int j = 0; j < outputSymbolsAlphabetSize; j++) {
                logEmiss[i, j] = Math.Log(Math.Max(ctxBest.emissionMatrix[i, j], 1e-100));
            }
        }

        // 2. Process each training trace using the Viterbi algorithm
        foreach (var trace in trainingTraces) {
            var seq = trace.list;
            int T = seq.Count;
            if (T == 0) continue;

            double[,] logV = new double[T, nStates];
            int[,] ptr = new int[T, nStates];

            // Initialization (t = 0)
            int obs0 = seq[0].observationState;
            for (int s = 0; s < nStates; s++) {
                // Assume uniform prior for the initial state
                logV[0, s] = Math.Log(1.0 / nStates) + logEmiss[s, obs0];
            }

            // Recursion (t = 1 to T - 1)
            for (int t = 1; t < T; t++) {
                int obs = seq[t].observationState;
                int actPrev = seq[t - 1].action; // The action that brought us to the current state

                for (int sNext = 0; sNext < nStates; sNext++) {
                    double maxLogProb = double.NegativeInfinity;
                    int bestPrevState = -1;

                    int idxNext = ctxBest.calcIdxOfStateAndAction(sNext, actPrev);

                    for (int sPrev = 0; sPrev < nStates; sPrev++) {
                        double prob = logV[t - 1, sPrev] + logTrans[sPrev, idxNext];
                        if (prob > maxLogProb) {
                            maxLogProb = prob;
                            bestPrevState = sPrev;
                        }
                    }

                    logV[t, sNext] = maxLogProb + logEmiss[sNext, obs];
                    ptr[t, sNext] = bestPrevState;
                }
            }

            // Termination & Backtracking
            int[] bestPath = new int[T];
            double bestFinalLogProb = double.NegativeInfinity;
            int bestFinalState = -1;

            for (int s = 0; s < nStates; s++) {
                if (logV[T - 1, s] > bestFinalLogProb) {
                    bestFinalLogProb = logV[T - 1, s];
                    bestFinalState = s;
                }
            }

            bestPath[T - 1] = bestFinalState;
            for (int t = T - 1; t > 0; t--) {
                bestPath[t - 1] = ptr[t, bestPath[t]];
            }

            if (verbosity >= 1) {
                Console.WriteLine($"Viterbi best path log-prob: {bestFinalLogProb}");
            }

            // Accumulate counts based on the optimal path (Hard EM)
            for (int t = 0; t < T; t++) {
                int s = bestPath[t];
                int obs = seq[t].observationState;
                emissionCounts[s, obs] += 1.0;

                if (t < T - 1) {
                    int sNext = bestPath[t + 1];
                    int act = seq[t].action;
                    int idxNext = ctxBest.calcIdxOfStateAndAction(sNext, act);
                    transitionCounts[s, idxNext] += 1.0;
                }
            }
        }

        // 3. Update ctxBest matrices (Maximization Step)
        for (int i = 0; i < nStates; i++) {
            for (int j = 0; j < nStates * nActions; j++) {
                ctxBest.transitionMatrix[i, j] = transitionCounts[i, j];
            }
            for (int j = 0; j < outputSymbolsAlphabetSize; j++) {
                ctxBest.emissionMatrix[i, j] = emissionCounts[i, j];
            }
        }

        CtxUtils.normalizeTransitionMatrix(ctxBest);
        CtxUtils.normalizeEmissionMatrix(ctxBest);
    }


    
    // /param observations  array with observations to be sampled/to learn
    // /param stateTrace    trace of states of this sampled trace
    public static double calcExpectation(List<TupleStateAndAction> observationAndActionList, List<int> stateTrace, Ctx ctx) {

        double prob = 1.0; // overall probability of the traced transitions and observations

        for (int idx = 0; idx < observationAndActionList.Count - 1; idx++) {
            int stateCurrent = stateTrace[idx];
            int stateNext = stateTrace[idx + 1];

            int idxNext = ctx.calcIdxOfStateAndAction(stateTrace[idx + 1], observationAndActionList[idx].action);

            int observation = observationAndActionList[idx].observationState;

            double transitionProbability = ctx.transitionMatrix[stateCurrent, idxNext];
            double emissionProbability = ctx.emissionMatrix[stateCurrent, observation];

            prob *= transitionProbability;
            prob *= emissionProbability;
        }

        return prob;
    }
    














    
    /// <summary>
    /// Finds the best single next action to take to reach the goal observation.
    /// </summary>
    public static int getBestNextAction(Ctx ctx, int startObservation, int goalObservation) {
        List<int> plan = planPath(ctx, startObservation, goalObservation);
        
        if (plan.Count > 0) {
            return plan[0]; // Return the very first action to take
        }
        
        return -1; // No path found
    }

    /// <summary>
    /// Computes the full optimal sequence of actions to reach a goal observation 
    /// from a starting observation using Dijkstra's algorithm on the latent graph.
    /// </summary>
    public static List<int> planPath(Ctx ctx, int startObservation, int goalObservation) {
        int nStates = ctx.retNumberOfStates();
        int nActions = ctx.nActions;

        double[] minDist = new double[nStates];
        int[] parentState = new int[nStates];
        int[] actionToReach = new int[nStates];
        bool[] visited = new bool[nStates];

        // 1. Initialize distances
        for (int i = 0; i < nStates; i++) {
            // If this hidden state (clone) emits our starting observation, it's a valid starting point.
            // We use > 0.5 because CSCG emissions become highly deterministic after training.
            if (ctx.emissionMatrix[i, startObservation] > 0.5) {
                minDist[i] = 0.0;
            } else {
                minDist[i] = double.PositiveInfinity;
            }
            parentState[i] = -1;
            actionToReach[i] = -1;
        }

        int bestGoalNode = -1;

        // 2. Dijkstra's Algorithm over the hidden states
        for (int iter = 0; iter < nStates; iter++) {
            // Find the unvisited node with the smallest distance
            int u = -1;
            double minD = double.PositiveInfinity;
            for (int i = 0; i < nStates; i++) {
                if (!visited[i] && minDist[i] < minD) {
                    minD = minDist[i];
                    u = i;
                }
            }

            // If all remaining nodes are unreachable, stop searching
            if (u == -1 || minD == double.PositiveInfinity) break;
            
            visited[u] = true;

            // Check if we reached a valid goal clone
            if (ctx.emissionMatrix[u, goalObservation] > 0.5) {
                bestGoalNode = u;
                break; // Found the shortest path to a goal state!
            }

            // Relax edges (evaluate all possible actions and next states)
            for (int a = 0; a < nActions; a++) {
                for (int v = 0; v < nStates; v++) {
                    int idxNext = ctx.calcIdxOfStateAndAction(v, a);
                    double transProb = ctx.transitionMatrix[u, idxNext];
                    
                    // Only consider transitions that the model has actually learned (> 0)
                    if (transProb > 1e-8) { 
                        // Convert probability to a positive distance weight
                        double weight = -Math.Log(transProb); 
                        
                        if (minDist[u] + weight < minDist[v]) {
                            minDist[v] = minDist[u] + weight;
                            parentState[v] = u;
                            actionToReach[v] = a; // The action taken to get from u to v
                        }
                    }
                }
            }
        }

        // 3. Backtrack to extract the sequence of actions
        List<int> actions = new List<int>();
        if (bestGoalNode == -1) {
            return actions; // No path found, return empty list
        }

        int curr = bestGoalNode;
        while (parentState[curr] != -1) {
            actions.Add(actionToReach[curr]);
            curr = parentState[curr];
        }

        // The actions were collected backwards (from goal to start), so reverse them
        actions.Reverse();
        return actions;
    }






}


// TRYING AREA
//
//
// here is a algorithm to sample a path over the hidden states with monte carlo to be able to update the probabilities of the transition probabilities and emission probabilities
public class Attempt0Entry {
    public void entry() {
        CscgOptimizer cscgOptimizer = new CscgOptimizer();


        cscgOptimizer.init();



        {
            TrainingTrace trainingTrace = new TrainingTrace();
            trainingTrace.list.Add(new TupleStateAndAction(0, 0));
            trainingTrace.list.Add(new TupleStateAndAction(2, 0));
            cscgOptimizer.trainingTraces.Add(trainingTrace);
        }


        cscgOptimizer.nOptimizationIterations = 1500;


        for (int optimizationIteration = 0; optimizationIteration < cscgOptimizer.nOptimizationIterations; optimizationIteration++) {
            cscgOptimizer.viterbiOptimizerStep();
        }


        int breakpointDebugHere7 = 1;
    }
}





















/*
public static class CustomActivations {
    public static Tensor Gelu(Tensor x) {
        var cdf = 0.5 * (1.0 + tf.tanh((tf.sqrt(new Tensor(2.0 / Math.PI)) * (x + 0.044715 * tf.pow(x, 3)))));
        return x * cdf;
    }
}
*/



public class Fnn {

    IModel model;
    public NDArray x_train, y_train;
    //NDArray x_test, y_test;



    public int nHiddenUnits = 5;



    // Generate using the NumPy-style random generator
    public NDArray np_matrix; // = np.random.normal(0.0f, 1.0f, new Shape(12, 12));

    int sizeInput;
    int sizeOutput;

    public int trainingBatchsize = 10;
    public int trainingEpochs = 3;

    public void setup(int sizeInput, int sizeOutput) {
        this.sizeInput = sizeInput;
        this.sizeOutput = sizeOutput;

        // CHECKME< is sizeInput and sizeOutput flipped here? >
        np_matrix = np.random.normal(0.0f, 1.0f, new Shape(sizeInput, sizeOutput));
    }

    public void PrepareData() {

        /*

        //(x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
        //x_train = x_train.reshape(new Shape(60000, 784)) / 255f;
        //x_test = x_test.reshape(new Shape(10000, 784)) / 255f;

        int here5 = 5;


        int countSamplesForTraining = 2; // how many samples are used for training.

        x_train = new NDArray(new Shape(countSamplesForTraining, 12), TF_DataType.TF_FLOAT);
        
        for(int idx = 0; idx < x_train.shape[1];idx++) {
            x_train[0, idx] = 0.0001f;
        }
        
        x_train[0, 5] = 0.5f;
        x_train[0, 6] = 0.2f;



        for (int idx = 0; idx < x_train.shape[1]; idx++)
        {
            x_train[1, idx] = 0.0001f;
        }

        x_train[1, 5] = 0.5f;
        x_train[1, 6] = 0.2f;


        ///y_train = x_train;



        y_train = new NDArray(new Shape(countSamplesForTraining, 12), TF_DataType.TF_FLOAT);

        for (int idx = 0; idx < y_train.shape[1]; idx++)
        {
            y_train[0, idx] = 0.0001f;
        }

        y_train[0, 7] = 0.5f;
        y_train[0, 9] = 0.2f;



        for (int idx = 0; idx < y_train.shape[1]; idx++)
        {
            y_train[1, idx] = 0.0001f;
        }

        y_train[1, 7] = 0.5f;
        y_train[1, 9] = 0.2f;
        */



        // Perform matrix multiplication (dot product)
        // In TF.NET NumPy, you can often use the * operator or np.dot
        //y_train = np.dot(x_train, np_matrix);
        //y_train = np.multiply(x_train, np_matrix); // not correct because it uses broadcast

        Tensor xTrainAsTfConstant = tf.constant(x_train);
        Tensor npMatrixAsTfConstant = tf.constant(np_matrix);

        y_train = tf.matmul(xTrainAsTfConstant, npMatrixAsTfConstant).numpy();

        //y_train = x_train @ np_matrix;

        int here6 = 5;
    }


    public void buildModel() {


        //int vocabSize = 10;
        // sizeOutput = vocabSize; // for example which uses crossentropy


        var inputs = keras.Input(shape: sizeInput);

        var layers = new LayersApi();

        var hiddenOutputs = layers.Dense(nHiddenUnits, activation: keras.activations.Relu).Apply(inputs);
        //var outputs = layers.Dense(64, activation: "gelu").Apply(inputs);
        //var outputs = layers.Dense(64, activation: keras.activations.Get("gelu")).Apply(inputs);
        //var outputs = layers.Dense(64, activation: tf.nn.elu).Apply(inputs);

        var outputs = layers.Dense(sizeOutput).Apply(hiddenOutputs);

        model = keras.Model(inputs, outputs, name: "model");
        model.summary();

        //model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
        //    optimizer: keras.optimizers.Adam(),
        //    metrics: new[] { "accuracy" });

        model.compile(loss: keras.losses.MeanSquaredError(),
            optimizer: keras.optimizers.AdamW(),
            metrics: new[] { "accuracy" });
    }

    public void train() {
        model.fit(x_train, y_train, batch_size: trainingBatchsize, epochs: trainingEpochs);
        //model.evaluate(x_test, y_test);

        int here7 = 5;
    }






    /// <summary>
    /// Performs a forward pass (inference) on a given Tensor stimulus without training.
    /// </summary>
    /// <param name="stimulus">The input Tensor. Expected shape: (batch_size, sizeInput)</param>
    /// <returns>The predicted output as a Tensor.</returns>
    public Tensor predict(Tensor stimulus) {
        if (model == null) {
            throw new InvalidOperationException("Model is not built. Please call buildModel() first.");
        }

        // model.Apply passes the Tensor directly through the computation graph.
        // setting training: false ensures that layers like Dropout or BatchNorm 
        // behave correctly for inference (evaluation) rather than training.
        Tensors outputTensors = model.Apply(stimulus, training: false);

        // model.Apply returns a 'Tensors' collection. 
        // Since your model has a single output layer, we return the first Tensor.
        return outputTensors[0];
    }




    /// <summary>
    /// Custom activation function equivalent to the Python version.
    /// It takes a Tensor and returns a Tensor with the operations applied.
    /// </summary>
    /// <param name="x">Input Tensor from the layer.</param>
    /// <returns>Output Tensor after applying the custom activation.</returns>
    /*
    public Tensor OutputActivation(Tensor x)
    {
        // K.switch(condition, then_expression, else_expression) is translated to tf.where(...)

        // Condition: x >= 0
        var condition = tf.greater_equal(x, 0);

        // Then expression: tf.math.tanh(x + 0.1) * 10
        var thenExpression = tf.math.tanh(x + 0.1f) * 10f;

        // Else expression: tf.math.tanh(x) + 1
        var elseExpression = tf.math.tanh(x) + 1f;

        return tf.where(condition, thenExpression, elseExpression);
    }
    */


}


public class ProgramA {
    public static void Main(string[] args) {
        Fnn fnn = new Fnn();
        fnn.setup(12, 12);
        fnn.PrepareData();
        fnn.buildModel();
        fnn.train();
    }
}



// class for datapoint
public class DatA {
    public NDArray x;
    public NDArray y;

    // TODO LOW : add creation time so we can use it to compute the age to be able to throw it out

    public DatA(NDArray x, NDArray y) {
        this.x = x;
        this.y = y;
    }
}






/*
// adapter for simple RFT experiment
// 
// is used to translate variable bindings to representation which is easier to work with for the NN+CSCG
//
//
public class RftAdapter {
    public RftAdapter() {
    }

    public void translate(int symbolA0, int symbolA1, int symbolB0, int symbolB1) {
        // TODO < find same binding and encode binding as special symbols >
    }
}
*/

















public class EnvState {
    public double SzX;
    public double SzY;
    public double BallX;
    public double BallY;
    public double BatX;
    public double BatVX;
    public double BatWidth;
    public double VirtualBatWidth;
    public double VX;
    public double VY;
    public double MulVX;
    public long Hits;
    public long Misses;
    public long T;
    public bool WasBallSpawned;
    public int Verbosity;

    // event reported back from simulation
    // 0 : no event
    // 1 : hit bat
    // 2 : missed bat
    public int eventFromSimulation;
}

public class PongSimulation {
    public static EnvState MakeEnvState() {
        double szX = 50.0;
        double szY = 20.0;

        return new EnvState
        {
            SzX = szX,
            SzY = szY,
            BallX = szX / 2,
            BallY = szY / 5,
            BatX = 0.0,
            BatVX = 0,
            BatWidth = 6,
            VirtualBatWidth = 6,
            VX = 1,
            VY = 1,
            MulVX = 0,
            Hits = 0,
            Misses = 0,
            T = 0,
            WasBallSpawned = false,
            Verbosity = 1,

            eventFromSimulation = 0,
        };
    }

    public static void SimStep(EnvState env, Random rng) {
        env.T += 1;

        // Wall collision logic for velocity
        if (env.BallX <= 0) {
            env.VX = 1;
        }
        else if (env.BallX >= env.SzX - 1) {
            env.VX = -1;
        }

        if (env.BallY <= 0) {
            env.VY = 1;
        }
        else if (env.BallY >= env.SzY - 1) {
            env.VY = -1;
        }


        // Ball movement
        /*
        if ((env.T % 2) == 1) {
            env.BallX += env.VX;
        }*/
        
        env.BallX += env.VX * env.MulVX;
        env.BallY += env.VY;

        // Bat collision logic (when ball reaches the top/bottom where the bat is)
        bool wasBatHit = false;
        if (env.BallY == 0) {
            if (Math.Abs(env.BallX - env.BatX) <= env.BatWidth) {
                wasBatHit = true;

                if (env.Verbosity > 0) {
                    Console.WriteLine("env: good");
                }
                env.eventFromSimulation = 1;
                env.Hits += 1;
            } else {
                if (env.Verbosity > 0) {
                    Console.WriteLine("env: bad");
                }
                env.eventFromSimulation = 2;
                env.Misses += 1;
            }
        }

        // Respawn logic
        env.WasBallSpawned = false;
        if (env.BallX < 0.0 && !wasBatHit) {
        // if (env.BallY == 0 || env.BallX == 0 || env.BallX >= env.SzX - 1) {
            // rng.Next(min, max) is exclusive of the upper bound, just like Rust's gen_range
            // Note: rng.Next requires int, so we cast the long values
            env.BallY = (env.SzY / 2) + rng.Next(0, (int)(env.SzY / 2));
            env.BallX = rng.Next(0, (int)env.SzX);
            env.VX = -1.0 + rng.NextDouble()*2.0;
            env.VY = -1.0 + rng.NextDouble()*2.0;
            env.WasBallSpawned = true;

            if (env.Verbosity > 0) {
                Console.WriteLine("env: respawn ball");
            }
        }

        // Bat movement logic
        //long h0 = Math.Min(env.SzX - 1 + env.BatWidth, env.BatX + env.BatVX * env.BatWidth / 2); // WRONG
        //env.BatX = Math.Max(env.BatWidth * 2, h0); // WRONG

        env.BatX += env.BatVX;

        env.BatX = Math.Max(env.BatX, env.BatWidth/2.0);
        env.BatX = Math.Min(env.BatX, env.SzX - env.BatWidth / 2.0);

        // Calculate and print ratio
        double ratio = (double)env.Hits;
        long totalAttempts = env.Hits + env.Misses;
        
        // Prevent division by zero on the first few frames
        if (totalAttempts > 0) {
            ratio /= totalAttempts;
        } else {
            ratio = 0.0; // Or double.NaN if you want to strictly match Rust's 0.0/0.0 behavior
        }

        if (env.Verbosity > 0) {
            Console.WriteLine($"PONG  Hits={env.Hits} misses={env.Misses} ratio={ratio} time={env.T}");
        }
    }
}


// training trace of state and action till goal was hit
public class TrainingTrace {
    public List<TupleStateAndAction> list = new List<TupleStateAndAction>();
}



// raw stimuli which was perceived by the agent over it's lifetime
public class PerceivedStimuli {
    public float[] obsArray;

    public PerceivedStimuli(float[] obsArray) {
        this.obsArray = obsArray;
    }
}



public class RunnerEnv {
    EnvState envState = PongSimulation.MakeEnvState();
    Random envRng = new Random();
    Random agentRng = new Random();

    TrainingTrace currentTrace = new TrainingTrace();

    List<TrainingTrace> trainingTraces = new List<TrainingTrace>();


    Fnn fnn;




   CscgOptimizer cscgOptimizer = new CscgOptimizer();

    // We want 3 discrete states (plus 1 reserved for the goal = 4 total symbols)
    int numDiscreteStates = 3;
    int obsVectorSize = 5; //// // BallX, BallY, BatX, VX, VY

    // Persistent random matrix for stable projection
    //NDArray projectionMatrix;



    // raw perceived stimuli over lifetime of agent
    List<PerceivedStimuli> rawPerceivedStimuliOverLifetime = new List<PerceivedStimuli>();


    public RunnerEnv() {
        // Initialize the random matrix ONCE using TF.NET's NumPy
        //projectionMatrix = np.random.normal(0.0f, 1.0f, new Shape(obsVectorSize, numDiscreteStates));


        fnn = new Fnn();

        fnn.setup(obsVectorSize, numDiscreteStates);

        // we need to build the model to be able to do forward pass for stimuli
        fnn.buildModel();
    }



    public void run() {


        long nEnvSteps = 15001;

        double motorbabblingChance = 0.1;
        double batVelocityMax = 0.1;

        // how many env steps do we make until we do retraining of CSCG?
        int cscgRetrainingPeriod = 500;


        // 0 : do nothing
        // 1 : -1 direction
        // 2 : +1 direction
        int actionIdx = 1;


        for (long envStep = 0; envStep < nEnvSteps; envStep++) {

            // FIXME LOW : we pause the world for retraining. We can't do that for anything realtime
            if (envStep % cscgRetrainingPeriod == 0 && envStep > 0) {
                // retrain CSCG

                Console.WriteLine("CSCG training: start training ...");

                cscgOptimizer.trainingTraces = trainingTraces;

                cscgOptimizer.init();
                for (int optimizationIteration = 0; optimizationIteration < cscgOptimizer.nOptimizationIterations; optimizationIteration++) {
                    cscgOptimizer.viterbiOptimizerStep();
                }

                Console.WriteLine("CSCG training: ... finished training");




                // retrain NN

                Console.WriteLine("CSCG training: start training ...");

                {
                    fnn.x_train = null; // set it to null to handle case when there is no training data

                    if (rawPerceivedStimuliOverLifetime.Count > 0) {

                        fnn.x_train = new NDArray(new Shape(rawPerceivedStimuliOverLifetime.Count, obsVectorSize), TF_DataType.TF_FLOAT);

                        for (int idxTrainingData = 0; idxTrainingData < rawPerceivedStimuliOverLifetime.Count; idxTrainingData++) {

                            for (int idxArr = 0; idxArr < fnn.x_train.shape[1]; idxArr++) {
                                fnn.x_train[idxTrainingData, idxArr] = rawPerceivedStimuliOverLifetime[idxTrainingData].obsArray[idxArr];
                            }

                        }
                    }

                }


                if (fnn.x_train != null) {
                    fnn.PrepareData();

                    fnn.buildModel();
                    fnn.train();
                }


                Console.WriteLine("CSCG training: finish training ...");

            }




            // pereceived state discretization
            int perceivedStateAsDiscrete = 0;
            {
                // 1. Extract raw state into a normalized float array
                // Normalizing by screen size (SzX, SzY) keeps the neural/matrix math stable
                float[] obsArray = new float[] {
                    (float)(envState.BallX / envState.SzX),
                    (float)(envState.BallY / envState.SzY),
                    (float)(envState.BatX / envState.SzX),
                    (float)envState.VX,
                    (float)envState.VY
                };



                // add to perceived stimuli over lifetime
                rawPerceivedStimuliOverLifetime.add(new PerceivedStimuli(obsArray));



                // Convert to a 1D Tensor (Shape: 1 x 5)
                NDArray obsVector = np.array(obsArray).reshape(new Shape(1, obsVectorSize));
                Tensor obsTensor = tf.constant(obsVector, dtype: tf.float32);
                
                
                




                /* OLD version with random matrix
                Tensor matrixTensor = tf.constant(projectionMatrix, dtype: tf.float32);

                // 2. Multiply vector by the random matrix to get Logits
                Tensor logits = tf.matmul(obsTensor, matrixTensor);
                */
                Tensor logits = fnn.predict(obsTensor);






                // 3. Apply Softmax to get probabilities
                Tensor probabilities = tf.nn.softmax(logits, axis: 1);

                // 4. Convert to a discrete value using Argmax
                // tf.argmax returns the index of the highest probability (0, 1, or 2)
                // Use tf.math.argmax instead of tf.argmax
                Tensor discreteStateTensor = tf.squeeze(tf.math.argmax(probabilities, axis: 1));

                // Note: argmax usually returns an int64 (long) in TensorFlow
                perceivedStateAsDiscrete = (int)(long)discreteStateTensor.numpy();
            }

            perceivedStateAsDiscrete = perceivedStateAsDiscrete + 1; // state 0 is reserved for the goal state

            // decision based on CSCG
            {
                if (cscgOptimizer.ctxBest != null) { // ctxBest can be null at the very beginning
                    int decisionActionIdx = CscgOptimizer.getBestNextAction(cscgOptimizer.ctxBest, perceivedStateAsDiscrete, 0);
                    if (decisionActionIdx >= 0) { // actionIdx=-1 happens early in training when the next action can't be determined, so we have to ignore this case
                        actionIdx = decisionActionIdx;
                        
                        Console.WriteLine($"CscgOptimizer decision making   actionIdx={actionIdx}");

                        int breakpointDebugHere9 = 1;
                    }
                }
            }


            // motor babbling
            if (agentRng.NextDouble() < motorbabblingChance) {
                if (agentRng.Next(2) == 0) {
                    actionIdx = 1;
                }
                else {
                    actionIdx = 2;
                }
            }




            // random control for testing
            bool enRandomControl = false;
            if (enRandomControl) {
                if (agentRng.Next(2) == 0) {
                    actionIdx = 1;
                }
                else {
                    actionIdx = 2;
                }
            }






            if (actionIdx == 1) {
                envState.BatVX = -batVelocityMax;
            }
            else if (actionIdx == 2) {
                envState.BatVX = batVelocityMax;
            }
            else if (actionIdx == 0) {
                envState.BatVX = 0;
            }

            currentTrace.list.Add(new TupleStateAndAction(perceivedStateAsDiscrete, actionIdx));




            envState.eventFromSimulation = 0; // we need to reset the event

            PongSimulation.SimStep(envState, envRng);

            if (envState.eventFromSimulation == 1) { // good

                // inject observation of goal state which is state=0 . we do this so that we can find the action to the goal-state
                currentTrace.list.Add(new TupleStateAndAction(0, 0));

                commitTraceAndResetTrace();
            }
            else if (envState.eventFromSimulation == 2) { // bad


                envState.BatX = new Random().NextDouble() * envState.SzX;


                resetTrace();
            }

        }

    }

    private void commitTraceAndResetTrace() {
        trainingTraces.Add(currentTrace);
        resetTrace();
    }

    private void resetTrace() {
        currentTrace = new TrainingTrace();
    }
}
































public class SkipConnectionModel : Model {
    private ILayer _dense1;
    private ILayer _dense2;
    private ILayer _outputLayer;

    public SkipConnectionModel(ModelArgs args) : base(args) {
        // Define layers
        _dense1 = keras.layers.Dense(64, activation: "relu");
        _dense2 = keras.layers.Dense(64, activation: "relu");
        _outputLayer = keras.layers.Dense(1); // Regression output
    }

    protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null) {
        // 1. Explicitly declare 'x' as a single Tensor
        Tensor x = _dense1.Apply(inputs);

        // 2. 'skip' is now also a single Tensor
        Tensor skip = x;

        // 3. Apply the second layer
        x = _dense2.Apply(x);

        // 4. Because both are now 'Tensor' (singular), the + operator will work perfectly!
        x = x + skip;

        // (Alternatively, you can always use the built-in math function: x = tf.add(x, skip); )

        return _outputLayer.Apply(x);
    }
}


class Program2 {
    static void Main(string[] args) {
        // Ensure eager execution is enabled (default in newer TF.NET, but good practice)
        tf.enable_eager_execution();

        Console.WriteLine("Generating dummy data...");
        // Generate dummy data: y = 2x + 1 + noise
        var np_x = np.random.uniform(-10f, 10f, new int[] { 1000, 10 });
        var np_y = np.sum(np_x, axis: 1)//, keepdims: true)
            * 2.0f + 1.0f;

        var x_train = tf.constant(np_x, dtype: tf.float32);
        var y_train = tf.constant(np_y, dtype: tf.float32);

        // 1. Instantiate the custom model
        var model = new SkipConnectionModel(new ModelArgs());

        // 2. Define the Optimizer
        var optimizer = keras.optimizers.Adam(learning_rate: 0.01f);

        int epochs = 50;
        int batchSize = 32;
        int numBatches = (int)x_train.shape[0] / batchSize;

        Console.WriteLine("Starting training loop...");

        // 3. Custom Training Loop
        for (int epoch = 0; epoch < epochs; epoch++) {
            float epochLoss = 0f;

            for (int b = 0; b < numBatches; b++) {
                // Slice the batch
                var x_batch = x_train[new Slice(b * batchSize, (b + 1) * batchSize)];
                var y_batch = y_train[new Slice(b * batchSize, (b + 1) * batchSize)];

                // Open a GradientTape to record operations
                using var tape = tf.GradientTape();

                // Forward pass
                var predictions = model.Apply(x_batch, training: true);

                // Compute custom loss
                var loss = CustomLoss(y_batch, predictions);

                // Compute gradients
                var gradients = tape.gradient(loss, model.TrainableVariables);

                // Apply gradients to update weights
                optimizer.apply_gradients(zip(gradients, model.TrainableVariables));

                epochLoss += (float)loss;
            }

            if ((epoch + 1) % 10 == 0) {
                Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {epochLoss / numBatches:F4}");
            }
        }

        Console.WriteLine("Training complete!");

        // Test the model
        var test_input = tf.constant(np.ones(new int[] { 1, 10 }, np.float32));
        var test_output = model.Apply(test_input, training: false);
        Console.WriteLine($"Test Input: {test_input.numpy()}");
        Console.WriteLine($"Test Output Prediction: {test_output.numpy()}");
    }

    // Custom Loss Function
    public static Tensor CustomLoss(Tensor y_true, Tensor y_pred) {
        var mse = tf.reduce_mean(tf.square(y_true - y_pred));
        var penalty = tf.reduce_mean(tf.abs(y_pred)) * 0.01f;
        return mse + penalty;
    }
}






