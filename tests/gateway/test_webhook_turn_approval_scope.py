"""AI Kosmo: a yolo webhook turn's approval bypass is the turn's alone: worker threads and kernel cells see it,
the hardline floor holds, and a reused Codex app-server session takes each turn's own policy."""
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from agent import codex_runtime
from gateway.config import Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_context import set_session_vars, clear_session_vars
from tools.approval import check_all_command_guards
from tools.approval_context import _get_approval_mode, unattended_turn
from tools.code_kernel import CellAuthority
from tools.thread_context import propagate_context_to_thread

COMMAND = 'rm -rf /tmp/kosmi-approval-scope/disposable-only'


def agent():
    return SimpleNamespace(_codex_session=None, session_cwd='/tmp/kosmi-approval-scope', tool_progress_callback=None,
                           _cached_system_prompt='fixed prompt', ephemeral_system_prompt=None)


class ApprovalScopeTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('hermes_cli.config.load_config_readonly', return_value={
            'approvals': {'mode': 'manual', 'unattended_mode': 'deny'}}))
        self.tokens = set_session_vars(platform='webhook', session_key='review-thread', session_id='same-session')
        self.addCleanup(clear_session_vars, self.tokens)

    def test_worker_cell_context_and_same_session_terminal_restoration(self):
        def decision():
            return (_get_approval_mode(), check_all_command_guards(COMMAND, 'local')['approved'])
        with unattended_turn(True):
            self.assertEqual(decision(), ('off', True))
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(propagate_context_to_thread(decision)).result(), ('off', True))
            cell = CellAuthority('same-session')
            self.assertEqual(cell.ctx.run(_get_approval_mode), 'off')
            self.assertFalse(check_all_command_guards('rm -rf /', 'local')['approved'])
        with unattended_turn(False):
            self.assertEqual(decision(), ('manual', False))

    def test_fresh_codex_session_follows_the_turn(self):
        with unattended_turn(True):
            a = agent()
            codex_runtime._ensure_codex_session(a)
            self.assertEqual(a._codex_session._decide_exec_approval({'command': COMMAND}), 'accept')
        with unattended_turn(False):
            b = agent()
            codex_runtime._ensure_codex_session(b)
            self.assertEqual(b._codex_session._decide_exec_approval({'command': COMMAND}), 'decline')

    def test_reused_codex_session_follows_the_next_non_yolo_turn(self):
        route = {'yolo': True}
        adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'routes': {'auto-demo': route}}))
        source = SessionSource(platform=Platform.WEBHOOK, chat_id='webhook:auto-demo:thread-review',
                               user_id='webhook:auto-demo')
        runner = SimpleNamespace(_adapter_for_source=lambda source: adapter)
        ctx = SimpleNamespace(source=source, session_id='same-session')
        a = agent()
        turn = TurnRunner(runner, ctx)
        def run_codex():
            codex_runtime._ensure_codex_session(a)
            return {'mode': _get_approval_mode(), 'exec': a._codex_session._decide_exec_approval({'command': COMMAND}),
                    'patch': a._codex_session._decide_apply_patch_approval({})}
        turn._run_sync_scoped = run_codex
        first = turn.run_sync()
        self.assertEqual(first, {'mode': 'off', 'exec': 'accept', 'patch': 'accept'})
        self.assertEqual(_get_approval_mode(), 'manual')
        route['yolo'] = False
        second = turn.run_sync()
        self.assertEqual(second, {'mode': 'manual', 'exec': 'decline', 'patch': 'decline'})


