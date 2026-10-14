import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments/sprint1_structural'))
from train import create_eval_env
from benchmark_bots import TacticalBot
from haxballgym.mutator import RandomStateMutator


class BenchmarkContract(unittest.TestCase):
    def test_tactical_styles_respect_team_frame_under_reflection(self):
        env=create_eval_env(16,1,1,str(ROOT/'maps/futsal_1v1.hbs'),8)
        env.state_mutator=RandomStateMutator(ball_speed=3.,rng=9876);env.reset();s=env.prev_state
        def reflect(a):
            a=a.copy();a[...,0]*=-1;return a
        mirror=dataclasses.replace(s,ball_pos=reflect(s.ball_pos),ball_vel=reflect(s.ball_vel),
                                   player_pos=reflect(s.player_pos[:,::-1]),player_vel=reflect(s.player_vel[:,::-1]))
        for style in TacticalBot.STYLES:
            b=TacticalBot(style)
            np.testing.assert_array_equal(b.act(s,[0]),b.act(mirror,[1]),err_msg=style)

    def test_iteration_watcher_stops_identified_process_and_saves_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'run';run.mkdir();live=run/'live.json'
            live.write_text(json.dumps(dict(status='running',completed_iterations=2)))
            trainer=root/'train.py'
            trainer.write_text("import json,signal,time,sys\nfrom pathlib import Path\np=Path(sys.argv[1])/'live.json'\ndef stop(*args):\n p.write_text(json.dumps(dict(status='completed',completed_iterations=3)))\n sys.exit(0)\nsignal.signal(signal.SIGTERM,stop)\np.write_text(json.dumps(dict(status='running',completed_iterations=3)))\nwhile True:time.sleep(.1)\n")
            process=subprocess.Popen([sys.executable,str(trainer),str(run)])
            try:
                subprocess.run([sys.executable,str(ROOT/'experiments/sprint1_structural/stop_at_iteration.py'),
                                '--run',str(run),'--pid',str(process.pid),'--iteration','3'],check=True,timeout=10)
                process.wait(timeout=3)
                record=json.loads((root/'iteration_stop.json').read_text())
                self.assertEqual(record['final_iteration'],3)
                self.assertEqual(record['status'],'exited')
            finally:
                if process.poll() is None:process.kill();process.wait()


if __name__=='__main__':unittest.main()
