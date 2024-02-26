import time
import heapq
import argparse
import logging
import json
import statistics
from datetime import datetime
from typing import List, Tuple, Any

# Global configuration file paths
EVENT_FILE = 'scheduled_events.json'
CONFIG_FILE = 'scheduler_config.json'

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

class PreciseScheduler:
    """
    A class representing a high-precision event scheduler.
    Events are scheduled with nanosecond precision.

    Attributes:
        events (List[Tuple[int, str, str]]): A heap of scheduled events.
    """

    def __init__(self):
        self.events: List[Tuple[int, str, str]] = []
        self.load_events()
        heapq.heapify(self.events)
        self.load_config()

    def add_event(self, name: str, delay_ns: int, action: str):
        """
        Schedule a new event.

        Args:
            name (str): The name of the event.
            delay_ns (int): The delay in nanoseconds until the event is triggered.
            action (str): The action to be executed as a string of Python code.
        """
        event_time_ns = time.perf_counter_ns() + delay_ns
        heapq.heappush(self.events, (event_time_ns, name, action))
        scheduled_time = self.convert_ns_to_datetime(event_time_ns)
        logging.info(f"Event '{name}' scheduled for {scheduled_time} ({event_time_ns} ns)")
        self.save_events()

    def load_events(self):
        """ Load scheduled events from a JSON file. """
        try:
            with open(EVENT_FILE, 'r') as file:
                self.events = json.load(file)
        except FileNotFoundError:
            logging.warning("Event file not found. Starting with an empty schedule.")

    def save_events(self):
        """ Save current events to a JSON file. """
        with open(EVENT_FILE, 'w') as file:
            json.dump(self.events, file)

    def start(self):
        """ Start executing scheduled events. """
        logging.info("Starting the scheduler...")
        while self.events:
            current_time = time.perf_counter_ns()
            event_time, name, action = self.events[0]
            if current_time >= event_time:
                heapq.heappop(self.events)
                self.execute_action(name, action, event_time)

    def execute_action(self, name: str, action: str, event_time: int):
        """
        Execute an action associated with an event.

        Args:
            name (str): The name of the event.
            action (str): The action code to execute.
            event_time (int): The scheduled time of the event in nanoseconds.
        """
        action_start_time = time.perf_counter_ns()
        # TODO: Replace 'exec' with a safer alternative
        exec(action)  # Execute the action
        action_end_time = time.perf_counter_ns()

        action_duration = action_end_time - action_start_time
        executed_time = self.convert_ns_to_datetime(action_end_time)
        error_margin = action_end_time - event_time
        logging.info(f"Event '{name}' executed at {executed_time} ({action_end_time} ns), "
                     f"Duration: {action_duration} ns, Margin of Error: {error_margin} ns")
        self.save_events()

    def benchmark_system(self, iterations: int = 100, delay_ns: int = 1000):
        """
        Benchmark the system to determine the average error margin.

        Args:
            iterations (int): Number of iterations for the benchmark.
            delay_ns (int): Delay in nanoseconds for each iteration.
        """
        logging.info("Starting benchmark...")
        temp_results = []
        for _ in range(iterations):
            start_time = time.perf_counter_ns()
            busy_wait(delay_ns)  # Wait for specified delay in ns
            end_time = time.perf_counter_ns()
            temp_results.append(end_time - start_time - delay_ns)

        filtered_results = self.filter_outliers(temp_results)
        avg_error = statistics.mean(filtered_results)
        logging.info(f"Average benchmark margin of error after filtering: {avg_error} ns")

        self.update_config(avg_error)
        return avg_error

    @staticmethod
    def filter_outliers(data: List[float], m: float = 2.0) -> List[float]:
        """
        Filter outliers from a dataset.

        Args:
            data (List[float]): The dataset to filter.
            m (float): The number of standard deviations to use as a threshold.
        
        Returns:
            List[float]: The filtered dataset.
        """
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        return [x for x in data if abs(x - mean) < m * std_dev]

    def update_config(self, avg_error: float):
        """
        Update the scheduler configuration based on benchmark results.

        Args:
            avg_error (float): The average error margin in nanoseconds.
        """
        config = {'avg_error_ns': avg_error}
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file)
        logging.info("Configuration updated based on benchmark results")

    def load_config(self):
        """ Load scheduler configuration from a JSON file. """
        try:
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                # Use the configuration to adjust scheduler settings
        except FileNotFoundError:
            logging.warning("No configuration file found, using default settings")

    @staticmethod
    def convert_ns_to_datetime(ns: int) -> datetime:
        """
        Convert a nanosecond timestamp to a datetime object.

        Args:
            ns (int): The timestamp in nanoseconds.

        Returns:
            datetime: The corresponding datetime object.
        """
        current_epoch_ns = int(datetime.now().timestamp() * 1e9)
        future_time_ns = current_epoch_ns + ns
        return datetime.fromtimestamp(future_time_ns / 1e9)
    
def busy_wait(duration_ns: int):
    """
    Perform a busy-wait for a specified duration in nanoseconds.

    Args:
        duration_ns (int): The duration in nanoseconds.
    """
    end_time = time.perf_counter_ns() + duration_ns
    while time.perf_counter_ns() < end_time:
        pass

def main():
    """ Main function to handle command-line arguments and start the scheduler. """
    scheduler = PreciseScheduler()
    parser = argparse.ArgumentParser(description="High-Precision Event Scheduler")
    parser.add_argument('--add-event', nargs=3, metavar=('NAME', 'DELAY_NS', 'ACTION'),
                        help="Add a new event to the scheduler")
    parser.add_argument('--benchmark', nargs='?', const=100, type=int,
                        help="Run a benchmark to determine system precision")
    args = parser.parse_args()

    if args.add_event:
        name, delay_ns, action = args.add_event
        scheduler.add_event(name, int(delay_ns), action)
    elif args.benchmark:
        scheduler.benchmark_system(args.benchmark)
    else:
        scheduler.start()

if __name__ == "__main__":
    main()
