import os
import sys
import time
from termcolor import colored
from datetime import timedelta

__all__ = ['ProgressBar']

def format_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    time_str = ''
    if h > 0:
        time_str += '{}hr'.format(int(h))
    if m > 0:
        time_str += '{}m'.format(int(m))
    ms = (s - int(s)) * 1000
    if int(s) > 0: 
        time_str += '{}s'.format(int(s))
    ms = (s - int(s)) * 1000
    time_str += '{}ms'.format(int(ms))
    return time_str

class ProgressBar:
    arrow_chr = colored(chr(187), color='cyan')
    covered_chr = colored(u'\u2550', color='red')
    def __init__(self, bar_proportion=0.4):
        self.bar_proportion = bar_proportion

    @property
    def terminal_length(self):
        return os.get_terminal_size().columns

    @property
    def bar_length(self):
        return int(self.terminal_length * self.bar_proportion)

    @property
    def start_time(self):
        if not hasattr(self, '_start_time'):
            print('\n')
            self._start_time = self._last_time = time.time()
        return self._start_time

    def step(self, index, total, msg=''):
        index = min(index, total - 1)
        bar_length = self.bar_length
        progress = int(bar_length * index / total)
        remaining = int(bar_length - progress) - 1
        
        sys.stdout.write('$[')
        progress_bar = self.covered_chr * progress + self.arrow_chr
        sys.stdout.write(progress_bar)
        remainder_bar = chr(215) * remaining
        sys.stdout.write(remainder_bar)
        sys.stdout.write(']')

        start_time = self.start_time
        current_time = time.time()
        step_time = current_time - self._last_time
        self._last_time = current_time
        total_time = current_time - start_time
        out = []
        out.append(' ( Total %s | Iter %s )' % (format_time(total_time), format_time(step_time)))
        if len(msg):
            out.append(' | ' + msg)
        
        msg = ''.join(out)
        sys.stdout.write(msg)
        for i in range(self.terminal_length - bar_length - len(msg) - 3):
            sys.stdout.write(' ')
        for i in range(self.terminal_length - int(bar_length / 2)):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (index + 1, total))

        if index < total:
            sys.stdout.write('\r')
        sys.stdout.flush()


if __name__ == '__main__':
    pbar = ProgressBar(0.4)

    for i in range(10000):
        for j in range(50000):
            continue
        pbar.step(i, 10000, msg='{} %'.format(round(100 * i / 10000, 5)))
    print('\n')

