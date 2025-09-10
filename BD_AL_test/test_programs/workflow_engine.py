"""
Advanced Workflow Engine with Complex Business Rules
Target: 30-50% coverage, ~80 cyclomatic complexity
"""

import time
import threading
from typing import List, Dict, Any, Optional, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"

class TaskType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    SUBPROCESS = "subprocess"
    HUMAN_TASK = "human_task"
    SERVICE_CALL = "service_call"
    SCRIPT_EXECUTION = "script_execution"

class ConditionOperator(Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class WorkflowVariable:
    name: str
    value: Any
    type_hint: str = "any"
    read_only: bool = False
    encrypted: bool = False
    
@dataclass
class TaskDefinition:
    task_id: str
    task_type: TaskType
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay: int = 1
    priority: int = 0
    
@dataclass
class TaskInstance:
    task_id: str
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error_message: Optional[str] = None
    retry_attempts: int = 0
    variables: Dict[str, WorkflowVariable] = field(default_factory=dict)

class ConditionEvaluator:
    def __init__(self):
        self.function_registry: Dict[str, Callable] = {}
        
    def register_function(self, name: str, function: Callable):
        """Register custom function for condition evaluation"""
        self.function_registry[name] = function
    
    def evaluate_condition(self, condition: Dict[str, Any], variables: Dict[str, WorkflowVariable]) -> bool:
        """Evaluate a single condition against workflow variables"""
        try:
            condition_type = condition.get('type', 'simple')
            
            if condition_type == 'simple':
                return self._evaluate_simple_condition(condition, variables)
            elif condition_type == 'compound':
                return self._evaluate_compound_condition(condition, variables)
            elif condition_type == 'custom_function':
                return self._evaluate_custom_function(condition, variables)
            else:
                return False
                
        except Exception:
            return False
    
    def _evaluate_simple_condition(self, condition: Dict[str, Any], variables: Dict[str, WorkflowVariable]) -> bool:
        """Evaluate simple condition (variable operator value)"""
        variable_name = condition.get('variable')
        operator = condition.get('operator')
        expected_value = condition.get('value')
        
        if variable_name not in variables:
            return False
        
        actual_value = variables[variable_name].value
        
        try:
            if operator == ConditionOperator.EQUALS.value:
                return actual_value == expected_value
            elif operator == ConditionOperator.NOT_EQUALS.value:
                return actual_value != expected_value
            elif operator == ConditionOperator.GREATER_THAN.value:
                return float(actual_value) > float(expected_value)
            elif operator == ConditionOperator.LESS_THAN.value:
                return float(actual_value) < float(expected_value)
            elif operator == ConditionOperator.CONTAINS.value:
                return str(expected_value) in str(actual_value)
            else:
                return False
                
        except (ValueError, TypeError):
            return False
    
    def _evaluate_compound_condition(self, condition: Dict[str, Any], variables: Dict[str, WorkflowVariable]) -> bool:
        """Evaluate compound condition (AND, OR, NOT)"""
        operator = condition.get('operator')
        sub_conditions = condition.get('conditions', [])
        
        if operator == ConditionOperator.AND.value:
            return all(self.evaluate_condition(sub_cond, variables) for sub_cond in sub_conditions)
        elif operator == ConditionOperator.OR.value:
            return any(self.evaluate_condition(sub_cond, variables) for sub_cond in sub_conditions)
        elif operator == ConditionOperator.NOT.value:
            if sub_conditions:
                return not self.evaluate_condition(sub_conditions[0], variables)
        
        return False
    
    def _evaluate_custom_function(self, condition: Dict[str, Any], variables: Dict[str, WorkflowVariable]) -> bool:
        """Evaluate custom function condition"""
        function_name = condition.get('function')
        parameters = condition.get('parameters', {})
        
        if function_name not in self.function_registry:
            return False
        
        try:
            # Resolve variable references in parameters
            resolved_params = {}
            for key, value in parameters.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    var_name = value[2:-1]
                    if var_name in variables:
                        resolved_params[key] = variables[var_name].value
                    else:
                        resolved_params[key] = value
                else:
                    resolved_params[key] = value
            
            return bool(self.function_registry[function_name](resolved_params))
            
        except Exception:
            return False

class WorkflowEngine:
    def __init__(self, max_concurrent_workflows: int = 10):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.active_workflows: Dict[str, 'WorkflowInstance'] = {}
        self.task_definitions: Dict[str, TaskDefinition] = {}
        self.condition_evaluator = ConditionEvaluator()
        self.global_variables: Dict[str, WorkflowVariable] = {}
        
        # Execution monitoring
        self.execution_stats: Dict[str, Any] = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'tasks_executed': 0,
            'average_execution_time': 0.0
        }
        
        # Thread management
        self.thread_pool: List[threading.Thread] = []
        self.task_queue = deque()
        self.running = False
        self.lock = threading.RLock()
        
    def register_task_definition(self, task_def: TaskDefinition) -> bool:
        """Register a task definition"""
        if not task_def.task_id:
            return False
        
        # Validate task definition
        if not self._validate_task_definition(task_def):
            return False
        
        self.task_definitions[task_def.task_id] = task_def
        return True
    
    def start_workflow(self, workflow_definition: Dict[str, Any], input_variables: Dict[str, Any] = None) -> Optional[str]:
        """Start a new workflow instance"""
        with self.lock:
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                return None
            
            # Generate workflow ID
            workflow_id = f"workflow_{int(time.time() * 1000)}_{len(self.active_workflows)}"
            
            # Create workflow instance
            workflow_instance = WorkflowInstance(
                workflow_id, 
                workflow_definition, 
                self.task_definitions,
                self.condition_evaluator
            )
            
            # Initialize workflow variables
            if input_variables:
                for var_name, value in input_variables.items():
                    workflow_instance.set_variable(var_name, value)
            
            # Add global variables
            for var_name, global_var in self.global_variables.items():
                if var_name not in workflow_instance.variables:
                    workflow_instance.set_variable(var_name, global_var.value, global_var.type_hint)
            
            # Validate workflow
            validation_result = workflow_instance.validate()
            if not validation_result['valid']:
                return None
            
            self.active_workflows[workflow_id] = workflow_instance
            self.execution_stats['workflows_started'] += 1
            
            # Start execution
            self._schedule_workflow_execution(workflow_instance)
            
            return workflow_id
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        return {
            'workflow_id': workflow_id,
            'status': workflow.status,
            'current_task': workflow.current_task,
            'completed_tasks': len(workflow.completed_tasks),
            'total_tasks': len(workflow.task_instances),
            'start_time': workflow.start_time,
            'variables': {name: var.value for name, var in workflow.variables.items()},
            'execution_path': workflow.execution_path
        }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        with self.lock:
            if workflow_id not in self.active_workflows:
                return False
            
            workflow = self.active_workflows[workflow_id]
            workflow.cancel()
            
            return True
    
    def suspend_workflow(self, workflow_id: str) -> bool:
        """Suspend a running workflow"""
        with self.lock:
            if workflow_id not in self.active_workflows:
                return False
            
            workflow = self.active_workflows[workflow_id]
            return workflow.suspend()
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a suspended workflow"""
        with self.lock:
            if workflow_id not in self.active_workflows:
                return False
            
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.SUSPENDED:
                workflow.resume()
                self._schedule_workflow_execution(workflow)
                return True
            
            return False
    
    def _validate_task_definition(self, task_def: TaskDefinition) -> bool:
        """Validate task definition"""
        # Check required fields
        if not task_def.name or not task_def.task_type:
            return False
        
        # Validate dependencies
        for dep_id in task_def.dependencies:
            if dep_id not in self.task_definitions and dep_id != task_def.task_id:
                continue  # Allow forward references
        
        # Validate conditions
        for condition in task_def.conditions:
            if not isinstance(condition, dict) or 'type' not in condition:
                return False
        
        # Validate parameters based on task type
        if task_def.task_type == TaskType.LOOP:
            if 'iterations' not in task_def.parameters and 'condition' not in task_def.parameters:
                return False
        elif task_def.task_type == TaskType.CONDITIONAL:
            if 'condition' not in task_def.parameters or 'true_path' not in task_def.parameters:
                return False
        
        return True
    
    def _schedule_workflow_execution(self, workflow_instance: 'WorkflowInstance'):
        """Schedule workflow for execution"""
        execution_thread = threading.Thread(
            target=self._execute_workflow,
            args=(workflow_instance,)
        )
        execution_thread.daemon = True
        execution_thread.start()
    
    def _execute_workflow(self, workflow_instance: 'WorkflowInstance'):
        """Execute workflow in separate thread"""
        try:
            workflow_instance.execute()
            
            with self.lock:
                if workflow_instance.status == WorkflowStatus.COMPLETED:
                    self.execution_stats['workflows_completed'] += 1
                elif workflow_instance.status == WorkflowStatus.FAILED:
                    self.execution_stats['workflows_failed'] += 1
                
                # Calculate average execution time
                if workflow_instance.end_time and workflow_instance.start_time:
                    execution_time = workflow_instance.end_time - workflow_instance.start_time
                    current_avg = self.execution_stats['average_execution_time']
                    completed = self.execution_stats['workflows_completed']
                    
                    if completed > 0:
                        self.execution_stats['average_execution_time'] = (
                            (current_avg * (completed - 1) + execution_time) / completed
                        )
                
        except Exception as e:
            workflow_instance.status = WorkflowStatus.FAILED
            workflow_instance.error_message = str(e)
    
    def cleanup_completed_workflows(self, older_than_minutes: int = 60):
        """Clean up old completed workflows"""
        current_time = time.time()
        cutoff_time = current_time - (older_than_minutes * 60)
        
        workflows_to_remove = []
        
        with self.lock:
            for workflow_id, workflow in self.active_workflows.items():
                if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                    workflow.end_time and workflow.end_time < cutoff_time):
                    workflows_to_remove.append(workflow_id)
            
            for workflow_id in workflows_to_remove:
                del self.active_workflows[workflow_id]
        
        return len(workflows_to_remove)

class WorkflowInstance:
    def __init__(self, workflow_id: str, definition: Dict[str, Any], 
                 task_definitions: Dict[str, TaskDefinition],
                 condition_evaluator: ConditionEvaluator):
        self.workflow_id = workflow_id
        self.definition = definition
        self.task_definitions = task_definitions
        self.condition_evaluator = condition_evaluator
        
        self.status = WorkflowStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_task: Optional[str] = None
        self.error_message: Optional[str] = None
        
        # Task management
        self.task_instances: Dict[str, TaskInstance] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.execution_path: List[str] = []
        
        # Variables and state
        self.variables: Dict[str, WorkflowVariable] = {}
        self.suspended_tasks: Set[str] = set()
        
        # Execution control
        self.should_cancel = False
        self.lock = threading.Lock()
        
    def validate(self) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        
        # Check required fields
        if 'tasks' not in self.definition:
            errors.append("Missing 'tasks' in workflow definition")
        
        if 'start_task' not in self.definition:
            errors.append("Missing 'start_task' in workflow definition")
        
        # Validate task references
        tasks = self.definition.get('tasks', [])
        task_ids = {task.get('task_id') for task in tasks if isinstance(task, dict) and 'task_id' in task}
        
        start_task = self.definition.get('start_task')
        if start_task not in task_ids:
            errors.append(f"Start task '{start_task}' not found in task list")
        
        # Validate task dependencies
        for task in tasks:
            if isinstance(task, dict) and 'dependencies' in task:
                for dep in task['dependencies']:
                    if dep not in task_ids:
                        errors.append(f"Task dependency '{dep}' not found")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(tasks):
            errors.append("Circular dependencies detected")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def execute(self):
        """Execute the workflow"""
        with self.lock:
            if self.status != WorkflowStatus.PENDING:
                return
            
            self.status = WorkflowStatus.RUNNING
            self.start_time = time.time()
        
        try:
            # Initialize task instances
            self._initialize_task_instances()
            
            # Start execution from start task
            start_task_id = self.definition.get('start_task')
            if start_task_id:
                self._execute_task_chain(start_task_id)
            
            # Check final status
            with self.lock:
                if not self.should_cancel:
                    if len(self.failed_tasks) > 0:
                        self.status = WorkflowStatus.FAILED
                    else:
                        self.status = WorkflowStatus.COMPLETED
                else:
                    self.status = WorkflowStatus.CANCELLED
                
                self.end_time = time.time()
                
        except Exception as e:
            with self.lock:
                self.status = WorkflowStatus.FAILED
                self.error_message = str(e)
                self.end_time = time.time()
    
    def _initialize_task_instances(self):
        """Initialize task instances from workflow definition"""
        tasks = self.definition.get('tasks', [])
        
        for task_config in tasks:
            if not isinstance(task_config, dict) or 'task_id' not in task_config:
                continue
            
            task_id = task_config['task_id']
            instance_id = f"{self.workflow_id}_{task_id}_{int(time.time() * 1000)}"
            
            task_instance = TaskInstance(
                task_id=task_id,
                instance_id=instance_id,
                workflow_id=self.workflow_id,
                status=WorkflowStatus.PENDING
            )
            
            # Copy workflow variables to task instance
            task_instance.variables = self.variables.copy()
            
            self.task_instances[task_id] = task_instance
    
    def _execute_task_chain(self, task_id: str):
        """Execute a chain of tasks starting from given task"""
        if self.should_cancel:
            return
        
        if task_id in self.completed_tasks or task_id in self.failed_tasks:
            return
        
        # Check dependencies
        if not self._check_task_dependencies(task_id):
            return
        
        # Execute current task
        success = self._execute_single_task(task_id)
        
        if not success:
            with self.lock:
                self.failed_tasks.add(task_id)
            return
        
        with self.lock:
            self.completed_tasks.add(task_id)
            self.execution_path.append(task_id)
        
        # Find and execute next tasks
        next_tasks = self._find_next_tasks(task_id)
        
        for next_task_id in next_tasks:
            if next_task_id not in self.completed_tasks:
                self._execute_task_chain(next_task_id)
    
    def _execute_single_task(self, task_id: str) -> bool:
        """Execute a single task"""
        if task_id not in self.task_instances:
            return False
        
        task_instance = self.task_instances[task_id]
        
        # Get task definition
        task_def = None
        for task_config in self.definition.get('tasks', []):
            if task_config.get('task_id') == task_id:
                task_def = task_config
                break
        
        if not task_def:
            return False
        
        with self.lock:
            task_instance.status = WorkflowStatus.RUNNING
            task_instance.start_time = time.time()
            self.current_task = task_id
        
        try:
            # Check pre-conditions
            conditions = task_def.get('conditions', [])
            if conditions and not self._evaluate_task_conditions(conditions):
                task_instance.status = WorkflowStatus.COMPLETED
                task_instance.result = "Skipped due to conditions"
                return True
            
            # Execute based on task type
            task_type = task_def.get('task_type', 'sequential')
            
            if task_type == TaskType.SEQUENTIAL.value:
                result = self._execute_sequential_task(task_def, task_instance)
            elif task_type == TaskType.PARALLEL.value:
                result = self._execute_parallel_task(task_def, task_instance)
            elif task_type == TaskType.CONDITIONAL.value:
                result = self._execute_conditional_task(task_def, task_instance)
            elif task_type == TaskType.LOOP.value:
                result = self._execute_loop_task(task_def, task_instance)
            elif task_type == TaskType.SERVICE_CALL.value:
                result = self._execute_service_call_task(task_def, task_instance)
            elif task_type == TaskType.SCRIPT_EXECUTION.value:
                result = self._execute_script_task(task_def, task_instance)
            elif task_type == TaskType.HUMAN_TASK.value:
                result = self._execute_human_task(task_def, task_instance)
            else:
                result = self._execute_default_task(task_def, task_instance)
            
            with self.lock:
                task_instance.status = WorkflowStatus.COMPLETED if result else WorkflowStatus.FAILED
                task_instance.end_time = time.time()
                task_instance.result = result
            
            return bool(result)
            
        except Exception as e:
            with self.lock:
                task_instance.status = WorkflowStatus.FAILED
                task_instance.error_message = str(e)
                task_instance.end_time = time.time()
            
            return False
    
    def _execute_conditional_task(self, task_def: Dict[str, Any], task_instance: TaskInstance) -> bool:
        """Execute conditional task"""
        condition = task_def.get('parameters', {}).get('condition')
        if not condition:
            return False
        
        # Evaluate condition
        condition_result = self.condition_evaluator.evaluate_condition(condition, self.variables)
        
        # Choose execution path
        if condition_result:
            true_path = task_def.get('parameters', {}).get('true_path', [])
            for subtask_id in true_path:
                if not self._execute_single_task(subtask_id):
                    return False
        else:
            false_path = task_def.get('parameters', {}).get('false_path', [])
            for subtask_id in false_path:
                if not self._execute_single_task(subtask_id):
                    return False
        
        return True
    
    def _execute_loop_task(self, task_def: Dict[str, Any], task_instance: TaskInstance) -> bool:
        """Execute loop task"""
        parameters = task_def.get('parameters', {})
        
        if 'iterations' in parameters:
            # Fixed iteration loop
            iterations = int(parameters['iterations'])
            subtasks = parameters.get('subtasks', [])
            
            for i in range(iterations):
                # Set loop variable
                self.set_variable('loop_index', i)
                
                for subtask_id in subtasks:
                    if not self._execute_single_task(subtask_id):
                        return False
                    
                    if self.should_cancel:
                        return False
        
        elif 'condition' in parameters:
            # Conditional loop
            condition = parameters['condition']
            subtasks = parameters.get('subtasks', [])
            max_iterations = parameters.get('max_iterations', 1000)
            
            iteration = 0
            while (iteration < max_iterations and 
                   self.condition_evaluator.evaluate_condition(condition, self.variables)):
                
                self.set_variable('loop_index', iteration)
                
                for subtask_id in subtasks:
                    if not self._execute_single_task(subtask_id):
                        return False
                    
                    if self.should_cancel:
                        return False
                
                iteration += 1
        
        return True
    
    def _execute_service_call_task(self, task_def: Dict[str, Any], task_instance: TaskInstance) -> bool:
        """Execute service call task (simulated)"""
        parameters = task_def.get('parameters', {})
        service_url = parameters.get('url', '')
        method = parameters.get('method', 'GET')
        timeout = parameters.get('timeout', 30)
        
        # Simulate service call delay
        time.sleep(min(timeout / 10, 1.0))  # Simulated network delay
        
        # Simulate success/failure based on URL
        if 'error' in service_url.lower():
            raise Exception(f"Service call failed to {service_url}")
        
        # Generate mock response
        mock_response = {
            'status': 'success',
            'data': f'Response from {service_url}',
            'timestamp': time.time()
        }
        
        # Store response in variables
        response_var = parameters.get('response_variable', 'service_response')
        self.set_variable(response_var, mock_response)
        
        return True
    
    # Additional helper methods
    def set_variable(self, name: str, value: Any, type_hint: str = "any"):
        """Set workflow variable"""
        with self.lock:
            self.variables[name] = WorkflowVariable(name, value, type_hint)
    
    def get_variable(self, name: str) -> Any:
        """Get workflow variable value"""
        return self.variables.get(name, WorkflowVariable(name, None)).value
    
    def cancel(self):
        """Cancel workflow execution"""
        with self.lock:
            self.should_cancel = True
            self.status = WorkflowStatus.CANCELLED
    
    def suspend(self) -> bool:
        """Suspend workflow execution"""
        with self.lock:
            if self.status == WorkflowStatus.RUNNING:
                self.status = WorkflowStatus.SUSPENDED
                return True
            return False
    
    def resume(self):
        """Resume workflow execution"""
        with self.lock:
            if self.status == WorkflowStatus.SUSPENDED:
                self.status = WorkflowStatus.RUNNING
    
    # More helper methods (simplified implementations)
    def _has_circular_dependencies(self, tasks: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies"""
        # Simplified cycle detection
        return False  # Placeholder
    
    def _check_task_dependencies(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        for task_config in self.definition.get('tasks', []):
            if task_config.get('task_id') == task_id:
                dependencies = task_config.get('dependencies', [])
                return all(dep in self.completed_tasks for dep in dependencies)
        return True
    
    def _find_next_tasks(self, completed_task_id: str) -> List[str]:
        """Find tasks that can be executed after completing given task"""
        next_tasks = []
        for task_config in self.definition.get('tasks', []):
            dependencies = task_config.get('dependencies', [])
            if completed_task_id in dependencies:
                task_id = task_config.get('task_id')
                if task_id and self._check_task_dependencies(task_id):
                    next_tasks.append(task_id)
        return next_tasks
    
    def _evaluate_task_conditions(self, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate all task conditions"""
        return all(self.condition_evaluator.evaluate_condition(condition, self.variables) 
                  for condition in conditions)
    
    # Simplified task execution methods
    def _execute_sequential_task(self, task_def, task_instance):
        return True
    
    def _execute_parallel_task(self, task_def, task_instance):
        return True
    
    def _execute_script_task(self, task_def, task_instance):
        return True
    
    def _execute_human_task(self, task_def, task_instance):
        return True
    
    def _execute_default_task(self, task_def, task_instance):
        return True

def workflow_engine_original(workflow_definition: Dict[str, Any], execution_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    # Validate workflow definition
    if not isinstance(workflow_definition, dict):
        return {'error': 'invalid_workflow_definition'}
    
    if not isinstance(execution_config, dict):
        return {'error': 'invalid_execution_config'}
    
    # Check required workflow fields
    required_fields = ['tasks', 'start_task']
    for field in required_fields:
        if field not in workflow_definition:
            return {'error': f'missing_required_field_{field}'}
    
    tasks = workflow_definition.get('tasks', [])
    if not tasks or not isinstance(tasks, list):
        return {'error': 'invalid_tasks_list'}
    
    # Validate tasks
    if len(tasks) > 100:  # Reasonable limit
        return {'error': 'too_many_tasks'}
    
    task_ids = set()
    for i, task in enumerate(tasks):
        if not isinstance(task, dict) or 'task_id' not in task:
            return {'error': f'invalid_task_at_index_{i}'}
        
        task_id = task['task_id']
        if task_id in task_ids:
            return {'error': f'duplicate_task_id_{task_id}'}
        
        task_ids.add(task_id)
    
    # Validate start task
    start_task = workflow_definition.get('start_task')
    if start_task not in task_ids:
        return {'error': 'invalid_start_task_reference'}
    
    # Extract execution configuration
    input_variables = execution_config.get('input_variables', {})
    timeout_seconds = execution_config.get('timeout_seconds', 60)
    
    if timeout_seconds <= 0 or timeout_seconds > 3600:
        return {'error': 'invalid_timeout_seconds'}
    
    # Validate input variables
    if not isinstance(input_variables, dict):
        return {'error': 'invalid_input_variables'}
    
    # Create and start workflow
    try:
        engine = WorkflowEngine(max_concurrent_workflows=5)
        
        # Register custom condition functions if needed
        def greater_than_threshold(params):
            return params.get('value', 0) > params.get('threshold', 100)
        
        engine.condition_evaluator.register_function('greater_than_threshold', greater_than_threshold)
        
        # Start workflow
        workflow_id = engine.start_workflow(workflow_definition, input_variables)
        
        if not workflow_id:
            return {'error': 'workflow_startup_failed'}
        
        # Monitor execution with timeout
        start_time = time.time()
        final_status = None
        
        while time.time() - start_time < timeout_seconds:
            status_info = engine.get_workflow_status(workflow_id)
            
            if not status_info:
                return {'error': 'workflow_status_unavailable'}
            
            current_status = status_info['status']
            
            if current_status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                final_status = current_status
                break
            
            time.sleep(0.1)  # Small delay for monitoring
        
        # Get final results
        final_status_info = engine.get_workflow_status(workflow_id)
        
        if not final_status_info:
            return {'error': 'final_status_unavailable'}
        
        # Analyze execution complexity
        completed_tasks = final_status_info['completed_tasks']
        total_tasks = final_status_info['total_tasks']
        execution_path = final_status_info.get('execution_path', [])
        
        if final_status == WorkflowStatus.COMPLETED:
            if completed_tasks == total_tasks:
                status = 'full_workflow_completion'
            elif completed_tasks > total_tasks * 0.8:
                status = 'mostly_completed'
            else:
                status = 'partially_completed'
        elif final_status == WorkflowStatus.FAILED:
            if completed_tasks == 0:
                status = 'early_failure'
            else:
                status = 'mid_execution_failure'
        elif final_status == WorkflowStatus.CANCELLED:
            status = 'workflow_cancelled'
        elif final_status is None:
            status = 'execution_timeout'
        else:
            status = 'unknown_completion_state'
        
        return {
            'status': status,
            'workflow_execution': final_status_info,
            'execution_metrics': {
                'completion_rate': completed_tasks / max(total_tasks, 1),
                'execution_path_length': len(execution_path),
                'unique_tasks_executed': len(set(execution_path)),
                'workflow_complexity': total_tasks
            },
            'engine_stats': engine.execution_stats
        }
        
    except Exception as e:
        return {'error': f'workflow_execution_exception: {str(e)}'}


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original target_function to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: target_function(workflow_definition, execution_config)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from target_function
    """
    # Set defaults for optional parameters based on function requirements
    if c is None:
        c = 0
    if d is None:
        d = 1000
    if b is None:
        b = -1000
    if c is None:
        c = 0
    if d is None:
        d = 1000
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-1000 <= param <= 1000):
                param = max(-1000, min(1000, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return target_function(a, b)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return target_function(a)
            except:
                pass
            return target_function(a)
        raise e
