// CST build/execute for rollbackable trial blocks:
//
//	attempt { <trial statements> } recover(e) { <fallback statements> }
//
// and for the explicit rejection statement:
//
//	reject "reason string"
//
// On normal completion (including break/continue/return, which preserve their
// control flow) the trial's changes are retained. On explicit rejection, on a
// DSL runtime failure, or on a statement runtime error raised inside a called
// function/subroutine, all mutable DSL state is restored to its pre-trial
// contents, the input stream is rewound to its pre-trial position, buffered
// stdout/stderr output is discarded, and the nearest recover block is run
// exactly once with e bound to the reason string.

package cst

import (
	"fmt"
	"os"

	"github.com/johnkerl/miller/v6/pkg/lib"
	"github.com/johnkerl/miller/v6/pkg/mlrval"
	"github.com/johnkerl/miller/v6/pkg/runtime"
	"github.com/johnkerl/miller/v6/pkg/types"
	"github.com/johnkerl/pgpg/go/lib/pkg/asts"
)

// ----------------------------------------------------------------
// Attempt node
// ----------------------------------------------------------------

type AttemptNode struct {
	attemptBlock   *StatementBlockNode
	reasonVariable *runtime.StackVariable
	recoverBlock   *StatementBlockNode
}

func (root *RootNode) BuildAttemptNode(astNode *asts.ASTNode) (*AttemptNode, error) {
	lib.InternalCodingErrorIf(astNode.Type != asts.NodeType(NodeTypeAttemptBlock))
	lib.InternalCodingErrorIf(astNode.Children == nil)
	lib.InternalCodingErrorIf(len(astNode.Children) != 3)

	attemptBlock, err := root.BuildStatementBlockNode(astNode.Children[0])
	if err != nil {
		return nil, err
	}

	nameNode := astNode.Children[1]
	lib.InternalCodingErrorIf(nameNode == nil || nameNode.Token == nil)
	reasonName := tokenLit(nameNode)
	if reasonName == "" {
		return nil, fmt.Errorf("attempt block: recover variable name is missing")
	}

	recoverBlock, err := root.BuildStatementBlockNode(astNode.Children[2])
	if err != nil {
		return nil, err
	}

	return &AttemptNode{
		attemptBlock:   attemptBlock,
		reasonVariable: runtime.NewStackVariable(reasonName),
		recoverBlock:   recoverBlock,
	}, nil
}

func (node *AttemptNode) Execute(state *runtime.State) (payload *BlockExitPayload, err error) {
	snapshot := state.BeginAttempt()

	caught := false
	reason := ""
	var rePanic any

	func() {
		defer func() {
			if r := recover(); r != nil {
				if rej, ok := r.(*runtime.AttemptReject); ok {
					// Explicit reject, guarded external effect, or a statement
					// runtime error raised inside a called function.
					caught = true
					reason = rej.Reason
					return
				}
				// Any other panic is an internal program fault: it is not a
				// branch rejection and must continue to propagate.
				rePanic = r
			}
		}()
		payload, err = node.attemptBlock.Execute(state)
	}()

	if rePanic != nil {
		state.AbandonAttemptWithoutRecover(snapshot)
		panic(rePanic)
	}

	if !caught && err == nil {
		// Normal completion. break/continue/return block exits are normal too:
		// the trial commits and the control-flow payload propagates to the
		// caller.
		outputs := state.CommitAttempt(snapshot)
		if publishErr := publishTrialOutputs(state, outputs); publishErr != nil {
			return nil, publishErr
		}
		return payload, nil
	}

	// Rejection or DSL runtime failure. A Go-level statement error carries its
	// message as the recover reason; MT_ERROR mlrvals are not errors here and
	// never trigger this path.
	if err != nil {
		reason = err.Error()
	}
	state.RollbackAttempt(snapshot)

	// Run the nearest recover block exactly once, with e bound to the reason
	// string in a frame scoped to the recover block. Errors or rejects raised
	// here propagate outward; this recover is not re-entered.
	state.Stack.PushStackFrame()
	defer state.Stack.PopStackFrame()
	if setErr := state.Stack.SetAtScope(node.reasonVariable, mlrval.FromString(reason)); setErr != nil {
		return nil, setErr
	}
	return node.recoverBlock.ExecuteFrameless(state)
}

// ----------------------------------------------------------------
// Reject node
// ----------------------------------------------------------------

type RejectNode struct {
	reasonExpression IEvaluable
}

func (root *RootNode) BuildRejectNode(astNode *asts.ASTNode) (*RejectNode, error) {
	lib.InternalCodingErrorIf(astNode.Type != asts.NodeType(NodeTypeRejectStatement))
	lib.InternalCodingErrorIf(astNode.Children == nil)
	lib.InternalCodingErrorIf(len(astNode.Children) != 1)

	reasonExpression, err := root.BuildEvaluableNode(astNode.Children[0])
	if err != nil {
		return nil, err
	}
	return &RejectNode{
		reasonExpression: reasonExpression,
	}, nil
}

func (node *RejectNode) Execute(state *runtime.State) (*BlockExitPayload, error) {
	reason := node.reasonExpression.Evaluate(state)
	if !reason.IsString() {
		// A non-string reason is itself a DSL runtime failure: if an attempt
		// block is active, it catches this error like any other; otherwise it
		// aborts execution as usual.
		return nil, fmt.Errorf(
			"reject: reason expression must evaluate to a string; got %s",
			reason.GetTypeName(),
		)
	}
	runtime.RaiseAttemptReject(reason.String())
	lib.InternalCodingErrorPanic("unreachable after reject")
	return nil, nil // unreachable
}

// ----------------------------------------------------------------
// External-effect guarding and trial output buffering helpers
// ----------------------------------------------------------------

// guardTrialExternalEffect rejects the current trial before an external
// effect (shell-out, file/pipe redirection, ENV write) can occur. Outside an
// attempt block the call is a no-op, preserving legacy behavior.
func guardTrialExternalEffect(state *runtime.State, what string) {
	if state.InTrial() {
		runtime.RaiseAttemptReject(fmt.Sprintf(
			"mlr: %s is not permitted within an attempt block", what,
		))
	}
}

// bufferTrialStdoutText retains unredirected stdout text for publication only
// if the outermost trial commits. Returns false when no trial is active, in
// which case the caller uses its legacy write path.
func bufferTrialStdoutText(state *runtime.State, text string) bool {
	if !state.InTrial() {
		return false
	}
	state.BufferTrialOutput(runtime.TrialOutput{
		Kind: runtime.TrialOutputStdoutText,
		Text: text,
	})
	return true
}

// bufferTrialStderrText retains unredirected stderr text for publication only
// if the outermost trial commits.
func bufferTrialStderrText(state *runtime.State, text string) bool {
	if !state.InTrial() {
		return false
	}
	state.BufferTrialOutput(runtime.TrialOutput{
		Kind: runtime.TrialOutputStderrText,
		Text: text,
	})
	return true
}

// bufferTrialDeferred retains an arbitrary legacy write (record-stream append,
// stdout/stderr handler-manager write, etc.) for replay at outermost commit.
func bufferTrialDeferred(state *runtime.State, flush func() error) bool {
	if !state.InTrial() {
		return false
	}
	state.BufferTrialOutput(runtime.TrialOutput{
		Kind:  runtime.TrialOutputDeferred,
		Flush: flush,
	})
	return true
}

// publishTrialOutputs flushes an outermost committed trial's buffered outputs
// in call order. Nested commits return nil outputs (they transfer to the
// parent frame instead).
func publishTrialOutputs(state *runtime.State, outputs []runtime.TrialOutput) error {
	for _, output := range outputs {
		switch output.Kind {
		case runtime.TrialOutputStdoutText:
			if state.OutputRecordsAndContexts != nil {
				*state.OutputRecordsAndContexts = append(
					*state.OutputRecordsAndContexts,
					types.NewOutputString(output.Text, state.Context),
				)
			} else {
				fmt.Print(output.Text)
			}
		case runtime.TrialOutputStderrText:
			fmt.Fprint(os.Stderr, output.Text)
		case runtime.TrialOutputDeferred:
			if err := output.Flush(); err != nil {
				return err
			}
		}
	}
	return nil
}

// ExternalEffectCallsiteNode wraps a state-free built-in function whose
// evaluation has an external effect (system, exec). Inside an attempt block
// the trial is rejected before the effect can happen; elsewhere evaluation is
// delegated unchanged.
type ExternalEffectCallsiteNode struct {
	inner IEvaluable
	what  string
}

func (node *ExternalEffectCallsiteNode) Evaluate(
	state *runtime.State,
) *mlrval.Mlrval {
	guardTrialExternalEffect(state, node.what)
	return node.inner.Evaluate(state)
}
