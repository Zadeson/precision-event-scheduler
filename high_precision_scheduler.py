# python high_precision_scheduler.py --benchmark --add-event "test_event" 5000000000 "print('Event triggered')" --start-scheduler

import time
import heapq
import argparse
import logging
import json
import statistics
from datetime import datetime, timedelta

# Global configuration file
EVENT_FILE = 'scheduled_events.json'
CONFIG_FILE = 'scheduler_config.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')

class PreciseScheduler:
    def __init__(self):
        self.events = []
        self.load_events()
        heapq.heapify(self.events)
        self.load_config()

    def add_event(self, name, delay_ns, action):
        event_time_ns = time.perf_counter_ns() + delay_ns
        heapq.heappush(self.events, (event_time_ns, name, action))
        scheduled_time = self.convert_ns_to_datetime(event_time_ns)
        logging.info(f"Event '{name}' scheduled for {scheduled_time} ({event_time_ns} ns)")
        self.save_events()

    def load_events(self):
        try:
            with open(EVENT_FILE, 'r') as file:
                self.events = json.load(file)
        except FileNotFoundError:
            self.events = []

    def save_events(self):
        with open(EVENT_FILE, 'w') as file:
            json.dump(self.events, file)

    def start(self):
        logging.info("Starting the scheduler...")
        while self.events:
            current_time = time.perf_counter_ns()
            event_time, name, action = self.events[0]
            if current_time >= event_time:
                heapq.heappop(self.events)
                action_start_time = time.perf_counter_ns()
                exec(action)  # Execute the action
                action_end_time = time.perf_counter_ns()
                action_duration = action_end_time - action_start_time

                executed_time = self.convert_ns_to_datetime(action_end_time)
                error_margin = action_end_time - event_time
                logging.info(f"Event '{name}' executed at {executed_time} ({action_end_time} ns), " +
                             f"Duration: {action_duration} ns, Margin of Error: {error_margin} ns")
                self.save_events()

    def benchmark_system(self, iterations=100, delay_ns=1000):
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

    def filter_outliers(self, data, m=2):
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        return [x for x in data if abs(x - mean) < m * std_dev]

    def update_config(self, avg_error):
        config = {'avg_error_ns': avg_error}
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file)
        logging.info("Configuration updated based on benchmark results")

    def load_config(self):
        try:
            with open(CONFIG_FILE, 'r') as file:
                config = json.load(file)
                # Use the configuration to adjust scheduler settings
        except FileNotFoundError:
            logging.info("No configuration file found, using default settings")

    def convert_ns_to_datetime(self, ns):
        current_epoch_ns = int(datetime.now().timestamp() * 1e9)
        future_time_ns = current_epoch_ns + ns
        return datetime.fromtimestamp(future_time_ns / 1e9)

def busy_wait(duration_ns):
    end_time = time.perf_counter_ns() + duration_ns
    while time.perf_counter_ns() < end_time:
        pass

def main():
    scheduler = PreciseScheduler()
    parser = argparse.ArgumentParser(description="High-Precision Event Scheduler")
    parser.add_argument('--add-event', nargs=3, metavar=('NAME', 'DELAY_NS', 'ACTION'))
    parser.add_argument('--start-scheduler', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()

    if args.benchmark:
        scheduler.benchmark_system()

    if args.add_event:
        delay_ns = int(args.add_event[1])
        scheduler.add_event(args.add_event[0], delay_ns, args.add_event[2])

    if args.start_scheduler:
        scheduler.start()

if __name__ == "__main__":
    main()
