import sys
sys.path.insert(0, '.')

from backend.nlp.context_loader import load_sprint_context, SprintContext
print('OK: context_loader imports')

from backend.nlp.jira_action_mapper import map_sprint_planning, map_pm_meeting_epics, map_standup_approval_payload
print('OK: jira_action_mapper imports')

from backend.nlp.pipeline_orchestrator import NLPOrchestrator
print('OK: pipeline_orchestrator imports')

from backend.db.crud import get_active_sprint, get_sprint_stories_with_details, get_sprint_assignments
print('OK: crud context functions exist')

result = map_sprint_planning(
    'Team commits to authentication epic',
    {'assignees': ['Alice'], 'estimates': ['80h'], 'dates': []},
    [{'story_id': 'SP-001', 'story_title': 'Auth story', 'actor': 'Alice',
      'action': 'assign_task', 'matched_via': 'db_assignment'}],
    sprint_meta={
        'sprint_name': 'Sprint 13', 'sprint_goal': 'Ship auth',
        'sprint_number': 13, 'capacity_hours': 80, 'velocity_target': 32,
        'velocity_actual': None, 'team_size': 4, 'committed_stories': []
    }
)
print('OK: map_sprint_planning sprint_name=' + result['sprint_name'] + ' capacity=' + str(result['capacity_hours']))

print()
print('All checks passed - context-aware pipeline is ready.')
