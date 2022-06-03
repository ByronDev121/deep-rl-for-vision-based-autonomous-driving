// Name: John M. McCullock
// Date: 11-21-06
// Description: Description:  Q-Learning Example 1

const qSize = 6;
const gamma = 0.8;
const iterations = 10;
var initialStates = new Array(1, 3, 5, 2, 4, 0);

var R = new Array(
  [-1, -1, -1, -1, 0, -1],
  [-1, -1, -1, 0, -1, 100],
  [-1, -1, -1, 0, -1, -1],
  [-1, 0, 0, -1, 0, -1],
  [0, -1, -1, 0, -1, 100],
  [-1, 0, -1, -1, 0, 100]
);

var Q = new Array(qSize);
for (a = 0; a < qSize; a++) {
  Q[a] = new Array(qSize);
}
var currentState;

function main() {
    console.log("running")
  let newWindow = window.open(
    "",
    "newwin",
    "height=500, width=800, toolbar=yes," +
      "scrollbars=yes, menubar=yes, location=yes," +
      "directories=yes, status=yes, copyhistory=yes," +
      "resizable=yes"
  );
  newWindow.document.write("<html>");
  newWindow.document.write("<title>Javascript Example Results</title>");
  newWindow.document.write("<body>");

  var newState;

  initialize();

  //Perform learning trials starting at all initial states.
  for (j = 0; j <= iterations - 1; j++) {
    for (i = 0; i <= qSize - 1; i++) {
      episode(initialStates[i]);
    } // i
  } // j

  //Print out Q matrix.
  for (i = 0; i <= qSize - 1; i++) {
    newWindow.document.write("<p>");
    for (j = 0; j <= qSize - 1; j++) {
      newWindow.document.write(Q[i][j]);
      if (j < qSize - 1) {
        newWindow.document.write(", ");
      }
    } // j
    newWindow.document.write("</p>");
  } // i
  newWindow.document.write("<p>&nbsp;</p>");

  //Perform tests, starting at all initial states.
  for (k = 0; k <= qSize - 1; k++) {
    currentState = initialStates[k];
    newState = 0;
    newWindow.document.write("<p>");
    do {
      newState = maximum(currentState, true);
      newWindow.document.write(currentState + ", ");
      currentState = newState;
    } while (currentState < 5);
    newWindow.document.write("5</p>");
  } // k

  newWindow.document.write("<p>&nbsp;</p>");
  newWindow.document.write("</body>");
  newWindow.document.write("</html>");
  newWindow.document.close();
}

function episode(initialState) {
  currentState = initialState;

  //Travel from state to state until goal state is reached.
  do {
    chooseAnAction();
  } while (currentState == 5);

  //When currentState = 5, run through the set once more to
  //for convergence.
  for (i = 0; i <= qSize - 1; i++) {
    chooseAnAction();
  } // i
}

function chooseAnAction() {
  var possibleAction = 0;

  //Randomly choose a possible action connected to the current state.
  possibleAction = getRandomAction(qSize, 0);

  if (R[currentState][possibleAction] >= 0) {
    Q[currentState][possibleAction] = reward(possibleAction);
    currentState = possibleAction;
  }
}

function getRandomAction(upperBound, lowerBound) {
  var action = 0;
  var choiceIsValid = false;
  var range = upperBound - lowerBound;

  //Randomly choose a possible action connected to the current state.
  do {
    //Get a random value between 0 and 6.
    action = lowerBound + Math.round(range * Math.random());

    if (R[currentState][action] > -1) {
      choiceIsValid = true;
    }
  } while (choiceIsValid == false);

  return action;
}

function initialize() {
  for (i = 0; i <= qSize - 1; i++) {
    for (j = 0; j <= qSize - 1; j++) {
      Q[i][j] = 0;
    } // j
  } // i
}

function maximum(state, returnIndexOnly) {
  // if returnIndexOnly = true, a Q matrix index is returned.
  // if returnIndexOnly = false, a Q matrix element is returned.

  var winner = 0;
  var foundNewWinner = false;
  var done = false;

  winner = 0;

  do {
    foundNewWinner = false;
    for (m = 0; m <= qSize - 1; m++) {
      if (m < winner || m > winner) {
        //Avoid self-comparison.
        if (Q[state][m] > Q[state][winner]) {
          winner = m;
          foundNewWinner = true;
        }
      }
    } // m

    if (foundNewWinner == false) {
      done = true;
    }
  } while (done == false);

  if (returnIndexOnly == true) {
    return winner;
  } else {
    return Q[state][winner];
  }
}

function reward(action) {
  return Math.round(R[currentState][action] + gamma * maximum(action, false));
}


main()