"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    this method add an penalty factor for move of opponent, for the current
    player, if his opponent has more move, his score will be decreased at a
    double speed. This method is inspired by diegoalejogm for his test on 
    various parameters of this penalty and 2 is the best case.
    
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player=player):
        return -math.inf
    if game.is_winner(player=player):
        return math.inf

    legal_moves = game.get_legal_moves(player=player)
    legal_moves_of_opponent = game.get_legal_moves(player=game.get_opponent(player=player))

    return float(len(legal_moves) - 2. * len(legal_moves_of_opponent))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    this is inspired by center_score function in sample_players.py for the logic
    is the closer player is in the center, the more score this player has.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player=player):
        return -math.inf
    if game.is_winner(player=player):
        return math.inf

    w, h = game.width / 2., game.height / 2.
    y_1, x_1 = game.get_player_location(player)
    y_2, x_2 = game.get_player_location(game.get_opponent(player))

    return float((h - y_1)**2 + (w - x_1)**2 - ((h - y_2)**2 + (w - x_2)**2))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player=player):
        return -math.inf
    if game.is_winner(player=player):
        return math.inf

    legal_moves = game.get_legal_moves(player)
    legal_moves_of_opponent = game.get_legal_moves(game.get_opponent(player))

    future_act = [(len(game.forecast_move(m).get_legal_moves(player)), m) for m in legal_moves]
    future_act_oppo = [(len(game.forecast_move(m).get_legal_moves(game.get_opponent(player))), m) for m in legal_moves_of_opponent]
    len_move, _ = max(future_act) if len(future_act) else (0, (-1, -1))
    len_move_oppo, _ = max(future_act_oppo) if len(future_act_oppo) else (0, (-1, -1))

    return float(len_move - len_move_oppo + len(legal_moves) - len(legal_moves_of_opponent))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
                      
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return -1, -1
        val = -math.inf, (-1, -1)
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            max_v = self.min_val(child_node, depth - 1)
            val = max(val, (max_v[0], move))
        return val[1]

    def max_val(self, game, depth):
        """

        :return: 
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return self.score(game, self), (-1, -1)  # add self to the score, it passes the functionality
        val = -math.inf, (-1, -1)
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            val = max(val, self.min_val(child_node, depth - 1))
        return val

    def min_val(self, game, depth):
        """
    
        :return: 
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:
            return self.score(game, self), (-1, -1)
        val = +math.inf, (-1, -1)
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            val = min(val, self.max_val(child_node, depth - 1))
        return val


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        depth = 1
        arr = [(-1, -1)]
        try:
            while True:
                arr.append(self.alphabeta(game, depth))
                depth += 1
        except SearchTimeout:
            pass

        return arr[-1]

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0 or not game.get_legal_moves():
            return -1, -1
        val = -math.inf
        b_move = -1, -1
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            val_tmp = self.min_alpha_beta(child_node, depth - 1, alpha, beta)
            if val_tmp > val:
                val, b_move = val_tmp, move
            alpha = max(alpha, val)
        return b_move

    def max_alpha_beta(self, game, depth, alpha, beta) -> float:
        """
 
        :return:         
        score : float
        
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:  # if there is no move, should I evaluate it or assign it an inf
            return self.score(game, self)
        val = float("-inf")
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            val_tmp = self.min_alpha_beta(child_node, depth - 1, alpha, beta)
            val = max(val_tmp, val)
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_alpha_beta(self, game, depth, alpha, beta) -> float:
        """
     
        :return: 
        score : float
        
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if depth == 0:  # if there is no move, should I evaluate it or assign it an inf
            return self.score(game, self)
        val = float("inf")
        for move in game.get_legal_moves():
            child_node = game.forecast_move(move)
            val_tmp = self.max_alpha_beta(child_node, depth - 1, alpha, beta)
            val = min(val_tmp, val)
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val
