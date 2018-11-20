import tensorflow as tf
from ANET import ANET, CaseManager
from HexState import HexState
import tflowtools as TFT
import numpy as np
from random import choice

class TOPP:
    def __init__(self, layerDims, hexSize, numberOfAgents, games, loadPath = "netsaver/topp/agent", verbose = False, bestk = 1):
        self.verbose = verbose
        self.hexSize = hexSize
        self.gamesNum = games
        self.bestk = bestk
        self.agents = self.createAgents(layerDims, loadPath, numberOfAgents)


    def createAgents(self, dims, path, numberOfAgents):
        agents = []
        for i in range(numberOfAgents):
            agents.append(HexAgent(dims, self.hexSize, path, i, self.bestk))
        return agents

    def playTournament(self):
        for i in range(len(self.agents)-1):
            for j in range(len(self.agents), i+1, -1):
                startingplayer = 1
                for _ in range(self.gamesNum):
                    self.playoutGame(self.agents[i], self.agents[j-1], startingplayer)
                    startingplayer = 3 - startingplayer

    def printResults(self):
        for agent in self.agents:
            print(agent.name+" won "+str(agent.wins)+" games")
    
    def playoutGame(self, agent1, agent2, startingplayer):
        game = HexState(1, self.hexSize)
        agents = [agent1, agent2]
        currentplayer = startingplayer
        if self.verbose: print(agent1.name+" vs "+agent2.name+", "+agents[startingplayer-1].name+" starts")
        while not game.isOver():
            move = agents[currentplayer-1].giveMoveFromState(game.getNeuralRepresentation())
            game.makeMove(move)
            currentplayer = 3 - currentplayer
            if self.verbose: game.printBoard()

        winner = game.getWinner()
        if startingplayer == 2:
            winner = 3-winner
        agents[winner-1].wins += 1
        if self.verbose: print(agents[winner-1].name+" wins")

    



class HexAgent:
    def __init__(self, layerDims, hexSize, loadPath, globalStep, bestk = 1):
        self.anet = None
        self.hexSize = hexSize
        self.name = "agent-"+str(globalStep)
        self.wins = 0
        self.bestk = bestk

        self.loadParams(layerDims, loadPath, globalStep)


    def loadParams(self, layerDims, loadPath, globalStep):
        self.anet = ANET(
        layer_dims = layerDims,
        softmax=True,
        case_manager = CaseManager([]))

        session = TFT.gen_initialized_session(dir="probeview")
        self.anet.current_session = session
        state_vars = []
        for m in self.anet.layer_modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.anet.state_saver = tf.train.Saver(state_vars)
        self.anet.state_saver.restore(self.anet.current_session, loadPath+"-"+str(globalStep))

    def giveMoveFromState(self, neuralRepresentation):
        feeder = {self.anet.input: [neuralRepresentation]}
        moves = self.anet.current_session.run(self.anet.output, feed_dict=feeder)
        legalMoves = [1] * (self.hexSize * self.hexSize)
        for i in range(0, len(neuralRepresentation)-2, 2):
            if neuralRepresentation[i] == 1 or neuralRepresentation[i+1] == 1:
                legalMoves[int(i/2)] = 0
        moves = moves * legalMoves
        #print(legalMoves)
        moves = moves[0]
        #print(moves)
        bestMoves = []
        for i in range(self.bestk):
            #print(max(moves))
            bestMove = np.where(moves == max(moves))[0][0]
            moves[bestMove] = 0
            [bestMoves.append(bestMove) for j in range((self.bestk-i)*(self.bestk-i))]
        bestMove = choice(bestMoves)
        return bestMove

def main():
    size = 5
    bk = 2

    topp = TOPP(layerDims=[size*size*2+2, size*size, size*size],
        hexSize = size,
        numberOfAgents = 5,
        games = 20,
        loadPath = "netsaver/topp5/agent",
        verbose = False,
        bestk = bk)

    """ agents = []
    for i in range(5):
        agents.append(HexAgent([size*size*2+2, size*size*2+2, size*size, size*size], 5, "netsaver/topp5random_2/agent", i, bk))
        agents[i].name = "agent-random_2-"+str(i)
    topp.agents += agents

    agents = []
    for i in range(5):
        agents.append(HexAgent([size*size*2+2, size*size*2+2, size*size, size*size], 5, "netsaver/topp5final/agent", i, bk))
        agents[i].name = "agent-final-"+str(i)
    topp.agents += agents
 """
    agents = []
    for i in range(5):
        agents.append(HexAgent([size*size*2+2, size*size*2+2, size*size, size*size], 5, "netsaver/topp5final2/agent", i, bk))
        agents[i].name = "agent-final2-"+str(i)
    topp.agents += agents

    topp.playTournament()

    topp.printResults()


if __name__ == '__main__':
    main()