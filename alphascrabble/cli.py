"""
Command-line interface for AlphaScrabble.

Provides commands for self-play, training, evaluation, and interactive play.
"""

import click
import os
import torch
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config, DEFAULT_CONFIG
from .nn.model import AlphaScrabbleNet
from .nn.train import Trainer
from .nn.evaluate import Evaluator
from .nn.dataset import TrainingDataManager
from .engine.movegen import MoveGenerator
from .engine.features import FeatureExtractor
from .engine.mcts import MCTS
from .lexicon.gaddag_loader import GaddagLoader
from .rules.board import Board, GameState
from .rules.tiles_en import TileBag
from .utils.logging import setup_logging, get_logger
from .utils.seeding import set_seed

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def main(ctx, config, seed, verbose):
    """AlphaScrabble: AlphaZero-style Scrabble engine."""
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    # Set seed if provided
    if seed is not None:
        set_seed(seed)
        DEFAULT_CONFIG.SEED = seed
    
    # Load config if provided
    if config:
        # TODO: Implement config loading from file
        pass
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = DEFAULT_CONFIG


@main.command()
@click.option('--games', '-g', default=10, help='Number of self-play games')
@click.option('--out', '-o', default='data/selfplay', help='Output directory')
@click.option('--policy-net', type=click.Path(exists=True), help='Policy network checkpoint')
@click.option('--simulations', '-s', default=160, help='MCTS simulations per move')
@click.option('--temperature', '-t', default=1.0, help='MCTS temperature')
@click.pass_context
def selfplay(ctx, games, out, policy_net, simulations, temperature):
    """Run self-play games to generate training data."""
    config = ctx.obj['config']
    logger.info(f"Starting self-play with {games} games")
    
    # Create output directory
    os.makedirs(out, exist_ok=True)
    
    # Initialize components
    try:
        # Load lexicon
        gaddag_loader = GaddagLoader(config.dawg_path, config.gaddag_path)
        if not gaddag_loader.load():
            console.print("[red]Failed to load lexicon files[/red]")
            return
        
        # Initialize move generator
        move_generator = MoveGenerator(gaddag_loader)
        feature_extractor = FeatureExtractor()
        
        # Load or create model
        if policy_net and os.path.exists(policy_net):
            model = AlphaScrabbleNet.load(policy_net)
            console.print(f"[green]Loaded model from {policy_net}[/green]")
        else:
            model = AlphaScrabbleNet()
            console.print("[yellow]Using untrained model[/yellow]")
        
        # Initialize MCTS
        mcts = MCTS(
            move_generator=move_generator,
            feature_extractor=feature_extractor,
            neural_net=model,
            simulations=simulations,
            temperature=temperature
        )
        
        # Run self-play games
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Playing self-play games...", total=games)
            
            for game_idx in range(games):
                # Play a single game
                game_result = _play_selfplay_game(mcts, config)
                
                # Save game data (simplified)
                game_file = os.path.join(out, f"game_{game_idx:04d}.pkl")
                # TODO: Implement game saving
                
                progress.update(task, advance=1)
        
        console.print(f"[green]Completed {games} self-play games[/green]")
        
    except Exception as e:
        logger.error(f"Self-play failed: {e}")
        console.print(f"[red]Self-play failed: {e}[/red]")


@main.command()
@click.option('--data', '-d', default='data/selfplay', help='Training data directory')
@click.option('--epochs', '-e', default=10, help='Number of training epochs')
@click.option('--batch-size', '-b', default=256, help='Batch size')
@click.option('--learning-rate', '-lr', default=0.001, help='Learning rate')
@click.option('--out', '-o', default='checkpoints', help='Output directory for checkpoints')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
@click.pass_context
def train(ctx, data, epochs, batch_size, learning_rate, out, resume):
    """Train the neural network."""
    config = ctx.obj['config']
    logger.info("Starting training")
    
    # Create output directory
    os.makedirs(out, exist_ok=True)
    
    try:
        # Initialize model
        if resume and os.path.exists(resume):
            model = AlphaScrabbleNet.load(resume)
            console.print(f"[green]Resumed training from {resume}[/green]")
        else:
            model = AlphaScrabbleNet()
            console.print("[yellow]Starting training from scratch[/yellow]")
        
        # Initialize data manager
        data_manager = TrainingDataManager(data)
        
        # Load training data
        # TODO: Implement data loading from self-play games
        train_games = []
        val_games = []
        
        if not train_games:
            console.print("[red]No training data found. Run self-play first.[/red]")
            return
        
        # Training configuration
        train_config = {
            'learning_rate': learning_rate,
            'weight_decay': 1e-4,
            'value_loss_weight': 1.0,
            'policy_loss_weight': 1.0,
            'l2_weight': 1e-4,
            'save_frequency': 5
        }
        
        # Initialize trainer
        trainer = Trainer(model, train_config, data_manager)
        
        # Train model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Training model...", total=epochs)
            
            history = trainer.train(train_games, val_games, epochs, batch_size)
            
            progress.update(task, completed=epochs)
        
        # Save final model
        final_model_path = os.path.join(out, "final_model.pt")
        model.save(final_model_path)
        
        console.print(f"[green]Training completed. Model saved to {final_model_path}[/green]")
        
        # Display training results
        _display_training_results(history)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        console.print(f"[red]Training failed: {e}[/red]")


@main.command()
@click.option('--net-a', type=click.Path(exists=True), help='First model checkpoint')
@click.option('--net-b', type=click.Path(exists=True), help='Second model checkpoint')
@click.option('--games', '-g', default=50, help='Number of evaluation games')
@click.option('--opponent', type=click.Choice(['random', 'greedy', 'model']), 
              default='random', help='Opponent type')
@click.pass_context
def eval(ctx, net_a, net_b, games, opponent):
    """Evaluate model performance."""
    config = ctx.obj['config']
    logger.info(f"Starting evaluation with {opponent} opponent")
    
    try:
        # Load lexicon
        gaddag_loader = GaddagLoader(config.dawg_path, config.gaddag_path)
        if not gaddag_loader.load():
            console.print("[red]Failed to load lexicon files[/red]")
            return
        
        # Initialize components
        move_generator = MoveGenerator(gaddag_loader)
        feature_extractor = FeatureExtractor()
        
        # Load model
        if not net_a or not os.path.exists(net_a):
            console.print("[red]Model checkpoint not found[/red]")
            return
        
        model = AlphaScrabbleNet.load(net_a)
        
        # Initialize evaluator
        eval_config = {
            'mcts_simulations': 160,
            'cpuct': 1.5,
            'temperature': 1.0
        }
        evaluator = Evaluator(model, move_generator, feature_extractor, eval_config)
        
        # Run evaluation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating model...", total=games)
            
            if opponent == 'random':
                results = evaluator.evaluate_against_random(games)
            elif opponent == 'greedy':
                results = evaluator.evaluate_against_greedy(games)
            elif opponent == 'model' and net_b:
                opponent_model = AlphaScrabbleNet.load(net_b)
                results = evaluator.evaluate_against_previous(opponent_model, games)
            else:
                console.print("[red]Invalid evaluation configuration[/red]")
                return
            
            progress.update(task, completed=games)
        
        # Display results
        _display_evaluation_results(results, opponent)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]Evaluation failed: {e}[/red]")


@main.command()
@click.option('--net', type=click.Path(exists=True), help='Model checkpoint')
@click.option('--human-first', is_flag=True, help='Human plays first')
@click.option('--simulations', '-s', default=160, help='MCTS simulations per move')
@click.pass_context
def play(ctx, net, human_first, simulations):
    """Play an interactive game against the bot."""
    config = ctx.obj['config']
    logger.info("Starting interactive game")
    
    try:
        # Load lexicon
        gaddag_loader = GaddagLoader(config.dawg_path, config.gaddag_path)
        if not gaddag_loader.load():
            console.print("[red]Failed to load lexicon files[/red]")
            return
        
        # Initialize components
        move_generator = MoveGenerator(gaddag_loader)
        feature_extractor = FeatureExtractor()
        
        # Load model
        if not net or not os.path.exists(net):
            console.print("[red]Model checkpoint not found[/red]")
            return
        
        model = AlphaScrabbleNet.load(net)
        
        # Initialize MCTS
        mcts = MCTS(
            move_generator=move_generator,
            feature_extractor=feature_extractor,
            neural_net=model,
            simulations=simulations
        )
        
        # Play interactive game
        _play_interactive_game(mcts, human_first)
        
    except Exception as e:
        logger.error(f"Interactive play failed: {e}")
        console.print(f"[red]Interactive play failed: {e}[/red]")


def _play_selfplay_game(mcts: MCTS, config: Config) -> dict:
    """Play a single self-play game."""
    # Initialize game
    board = Board()
    tile_bag = TileBag()
    
    players = ["Player1", "Player2"]
    scores = [0, 0]
    racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
    
    game_state = GameState(
        board=board,
        players=players,
        scores=scores,
        racks=racks,
        current_player=0,
        tile_bag=tile_bag
    )
    
    moves = []
    move_count = 0
    max_moves = 200
    
    while not game_state.game_over and move_count < max_moves:
        # Get best move from MCTS
        move = mcts.get_best_move(game_state)
        moves.append(move)
        
        # Apply move
        if move.tiles:  # Not a pass
            game_state.board.apply_move(move)
            game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
            game_state.add_score(game_state.current_player, move.total_score)
            game_state.draw_tiles(len(move.tiles))
        
        game_state.next_player()
        game_state.check_game_over()
        move_count += 1
    
    # Calculate final scores
    game_state.calculate_final_scores()
    
    return {
        'moves': moves,
        'scores': game_state.scores,
        'winner': game_state.winner,
        'game_length': move_count
    }


def _play_interactive_game(mcts: MCTS, human_first: bool):
    """Play an interactive game against the human."""
    # Initialize game
    board = Board()
    tile_bag = TileBag()
    
    players = ["Human", "Bot"]
    scores = [0, 0]
    racks = [tile_bag.draw_tiles(7), tile_bag.draw_tiles(7)]
    
    game_state = GameState(
        board=board,
        players=players,
        scores=scores,
        racks=racks,
        current_player=0 if human_first else 1,
        tile_bag=tile_bag
    )
    
    console.print("[bold blue]Welcome to AlphaScrabble![/bold blue]")
    console.print("Type 'help' for commands, 'quit' to exit")
    
    while not game_state.game_over:
        console.print(f"\n[bold]Current scores:[/bold] Human: {game_state.scores[0]}, Bot: {game_state.scores[1]}")
        console.print(f"[bold]Current player:[/bold] {game_state.players[game_state.current_player]}")
        
        # Display board
        console.print("\n[bold]Board:[/bold]")
        console.print(game_state.board.display())
        
        # Display current player's rack
        current_rack = game_state.get_current_rack()
        rack_str = " ".join(tile.display_letter for tile in current_rack)
        console.print(f"\n[bold]Your rack:[/bold] {rack_str}")
        
        if game_state.current_player == 0:  # Human's turn
            _handle_human_move(game_state, mcts)
        else:  # Bot's turn
            _handle_bot_move(game_state, mcts)
        
        game_state.next_player()
        game_state.check_game_over()
    
    # Game over
    console.print("\n[bold]Game Over![/bold]")
    console.print(f"Final scores: Human: {game_state.scores[0]}, Bot: {game_state.scores[1]}")
    
    if game_state.winner == 0:
        console.print("[green]Human wins![/green]")
    elif game_state.winner == 1:
        console.print("[red]Bot wins![/red]")
    else:
        console.print("[yellow]It's a tie![/yellow]")


def _handle_human_move(game_state: GameState, mcts: MCTS):
    """Handle human player's move."""
    while True:
        try:
            command = click.prompt("Enter move (or 'help', 'pass', 'quit')", type=str)
            
            if command.lower() == 'quit':
                game_state.game_over = True
                return
            elif command.lower() == 'help':
                _show_help()
                continue
            elif command.lower() == 'pass':
                # Pass move
                break
            else:
                # Parse move (simplified)
                # TODO: Implement proper move parsing
                console.print("[yellow]Move parsing not implemented yet[/yellow]")
                continue
                
        except (click.Abort, KeyboardInterrupt):
            game_state.game_over = True
            return


def _handle_bot_move(game_state: GameState, mcts: MCTS):
    """Handle bot's move."""
    console.print("\n[bold]Bot is thinking...[/bold]")
    
    # Get best move from MCTS
    move = mcts.get_best_move(game_state)
    
    if move.tiles:  # Not a pass
        console.print(f"[bold]Bot plays:[/bold] {move.notation}")
        
        # Apply move
        game_state.board.apply_move(move)
        game_state.remove_tiles_from_rack([tile.tile for tile in move.tiles])
        game_state.add_score(game_state.current_player, move.total_score)
        game_state.draw_tiles(len(move.tiles))
    else:
        console.print("[bold]Bot passes[/bold]")


def _show_help():
    """Show help information."""
    help_text = """
Available commands:
- Enter a move in format: WORD ROWCOL (e.g., "HELLO A8")
- 'pass' - Pass your turn
- 'help' - Show this help
- 'quit' - Quit the game
    """
    console.print(help_text)


def _display_training_results(history: dict):
    """Display training results."""
    table = Table(title="Training Results")
    table.add_column("Epoch", style="cyan")
    table.add_column("Train Loss", style="green")
    table.add_column("Val Loss", style="red")
    table.add_column("Learning Rate", style="yellow")
    
    for i, (train_loss, val_loss, lr) in enumerate(
        zip(history['train_loss'], history['val_loss'], history['learning_rate'])
    ):
        table.add_row(
            str(i + 1),
            f"{train_loss:.4f}",
            f"{val_loss:.4f}",
            f"{lr:.6f}"
        )
    
    console.print(table)


def _display_evaluation_results(results: dict, opponent: str):
    """Display evaluation results."""
    table = Table(title=f"Evaluation Results vs {opponent.title()}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Win Rate", f"{results['win_rate']:.2%}")
    table.add_row("Wins", str(results['wins']))
    table.add_row("Total Games", str(results['total_games']))
    
    console.print(table)


if __name__ == '__main__':
    main()
